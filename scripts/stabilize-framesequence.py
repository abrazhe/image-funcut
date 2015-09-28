#!/usr/bin/env python

from __future__ import division

import os
import argparse

import json

from functools import partial

import numpy as np
from scipy import ndimage,signal

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from imfun import opflowreg, atrous, fseq, lib

_smoothing_filters = {'gaussian': ndimage.gaussian_filter,
                      'atrous': atrous.smooth,
                      'median': signal.medfilt}




def main():
    parser = argparse.ArgumentParser(description="""Motion stabilize image stack (frame sequence)""")
    argdict =  {
        'imagestacks' : dict(nargs='+'),
        '-j': ('--json', dict(default=None, help="json file with default parameters")),
        '-p': ('--pretend', dict(action='store_true',help="Pretend mode: don't do anything, dry-run")),
        '-ch': dict(default='r', choices='rgb012', help='color channel to use'),
        '-s': ('--smooth', dict(action='append', metavar = ('FILTER', 'PARAMETER'), nargs='+',
                                help="smoothing filters; available filters: {median, gaussian, atrous}")),
        '-m': ('--model', dict(action='append', metavar = ("MODEL", "PARAMETER"),
                               nargs = '+',
                               #choices = ['shifts', 'softmesh', 'affine', 'Greenberg-Kerr'],
                               help='add movement model to use for stabilization;\
                               available models: {shifts, softmesh,  affine, Greenberg-Kerr, homography}')),
        '-t' : ('--type', dict(default='template',
                                  choices = ['template', 'recursive'],
                                  help='stabilization type')),
        '-n': ('--ncpu', dict(default=4, type=int, help="number of CPU cores to use")),
        '--record': dict(default=None, help='record within file to use (where applicable)'),
        '-v': ('--verbose', dict(action='count', help='increment verbosity level')),
        '--with-movies': dict(action='store_true'),
        '--suff': dict(default='', help="optional suffix to append to saved registration recipe"),
        '--fps': dict(default=25,type=float,help='fps of exported movie'),
        '--bitrate':dict(default=2000,type=float, help='bitrate of exported movie'),
        '--no-zstacks': dict(action='store_true', help='try to avoid z-stacks')
        }
    for arg,kw in argdict.items():
        if isinstance(kw, dict):
            parser.add_argument(arg,  **argdict[arg])
        else:
            parser.add_argument(arg, kw[0], **kw[1])

    args = parser.parse_args()

    if args.model is None:
        args.model = ['softmesh']
    if args.smooth is None:
        args.smooth = []
    else:
        for m in args.model:
            if not isinstance(m, basestring) and len(m)>1:
                params = json.loads(m[1])
                m[1] = params
    for smoother in args.smooth:
        if len(smoother) == 1:
            default_par = (smoother[0] in ['median']) and 3 or 1
            smoother.append(default_par)

    # override everything if json parameter is given
    if args.json :
        with open(args.json) as jsonfile:
            pars = json.load(jsonfile)
            for key,val in pars.items():
                setattr(args, key, val)
            
    if args.verbose > 2:
        print args

    registrators = opflowreg.RegistrationInterfaces

    def apply_reg(frames):
        if args.type == 'template':
            if args.verbose > 1:
                print 'stabilization type is template'
            tstart = len(frames)/2
            tstop = min(len(frames),tstart+50)
            template = np.max(frames[tstart:tstop],axis=0)
            def register_stack(stack, registrator, **fnargs):
                return opflowreg.register_stack_to_template(stack,template,registrator,njobs=args.ncpu,**fnargs)
        elif args.type == 'recursive':
            def register_stack(stack, registrator,**fnargs):
                return opflowreg.register_stack_recursive(stack,registrator,**fnargs)[1]
        else:
            raise NameError("Unknown registration type")
        # TODO: below is just crazy. has to be made neat later
        reg_dispatcher = {'affine':registrators.affine,
                          'homography':registrators.homography,
                          'shifts':registrators.shifts,
                          'Greenberg-Kerr':registrators.greenberg_kerr,
                          'softmesh':registrators.softmesh}
        operations = args.model
        newframes = frames
        warp_history = []
        for movement_model in operations:
            if not isinstance(movement_model, basestring):
                if len(movement_model)>1:
                    model, model_params = movement_model
                else:
                    model, model_params = movement_model[0],{}
            else:
                model = movement_model
                model_params = {}
            if args.verbose > 1:
                print 'correcting for {} with params: {}'.format(model, model_params)
            warps = register_stack(newframes, reg_dispatcher[model], **model_params)
            warp_history.append(warps)
            newframes = opflowreg.apply_warps(warps, newframes, njobs=args.ncpu)
        final_warps = [lib.flcompose(*warpchain) for warpchain in zip(*warp_history)]
        del newframes
        return final_warps
    
    for stackname in args.imagestacks:
        out_suff = make_outname_suffix(args)
        outname = stackname + out_suff + '.stab'
        try:
            if args.verbose:
                print '\nCalculating motion stabilization for file {}'.format(stackname)
                
            if args.pretend: continue

            fs = fseq.open_seq(stackname, ch=args.ch, record=args.record)

            if 'no_zstacks' and guess_fseq_type(fs) == 'Z':
                continue
            
            smoothers = get_smoothing_pipeline(args.smooth)
            fs.fns = smoothers
            warps = apply_reg(fs)
            opflowreg.save_recipe(warps, outname)
            if args.verbose:
                print 'saved motions stab recipe to {}'.format(outname)
            del fs
            
            if args.with_movies:

                if args.verbose>2:
                    print stackname+'-before-video.mp4'
                fsall = fseq.open_seq(stackname, ch='all', record=args.record)
                vl, vh = fsall.data_percentile(0.5), fsall.data_percentile(99.5)
                vl,vh = np.min(vl), np.max(vh)
                if args.verbose > 10:
                    print 'vl, vh: ', vl, vh

                proj1 = fsall.time_project(fn=partial(np.mean, axis=0))

                fs2 = opflowreg.apply_warps(warps, fsall, njobs=args.ncpu)
                proj2 = fs2.time_project(fn=partial(np.mean, axis=0))

                fig,axs = plt.subplots(1,2,figsize=(12,5.5))
                def _lutfn(f): return np.clip((f-vl)/(vh-vl), 0, 1)
                #def _lutfn(f): return np.dstack([np.clip(f[...,k],vl[k],vh[k])/vh[k] for k in range(f.shape[-1])])
                for ax,f,t in zip(axs,(proj1,proj2),['before','stabilized']):
                    ax.imshow(_lutfn(f),aspect='equal')
                    #imh = ax.imshow(f[...,0],aspect='equal',vmin=vl,vmax=vh); plt.colorbar(imh,ax=ax)
                    plt.setp(ax, xticks=[],yticks=[],frame_on=False)
                    ax.set_title(t)
                plt.savefig(stackname+out_suff+'-average-projections.png')
                fig.clf()
                plt.close(fig)

                if args.verbose > 2:
                    print stackname+out_suff+'-stabilized-video.mp4'

                fseq.to_movie([fsall, fs2],
                              stackname+out_suff+'-stabilized-video.mp4',
                              titles=['before', 'stabilized'],
                              bitrate=3000)

                del fsall, fs2
                
                plt.close('all')
            del warps
            if args.verbose>2: print 'Done'


        except Exception as e:
            print "Couldn't process {} becase  {}".format(stackname, e)
        
def guess_fseq_type(fs):
    dz, units = fs.meta['axes'][0]
    out = ''
    if units in ['um','mm','m','nm']:
        out = 'Z'
    elif units in ['ms','us', 's', 'msec', 'usec', 'ns', 'sec']:
        out = 'T'
    return out

def make_outname_suffix(args):
    models = [isinstance(m,basestring) and m or m[0] for m in args.model]
    ch = "ch_{}".format(args.ch)
    if args.smooth is not None:
        smoothers = ["{}_{}".format(name,par) for name,par in args.smooth]
    else:
        smoothers = []
    if args.json is not None:
        jsonname = os.path.split(args.json)[1]
        return ''.join(('-', '-'.join([jsonname, args.suff])))
    else:
        return ''.join(('-','-'.join([ch]+smoothers+[args.type]+models+[args.suff])))

def get_smoothing_pipeline(smooth_entry):
    if smooth_entry is None: return []
    pipeline = []
    for name, par in smooth_entry:
        if name not in _smoothing_filters:
            raise NameError('unknown blur filter name')
        smoother = _smoothing_filters[name]
        def _filter(f): return smoother(f.astype(np.float), float(par))
        pipeline.append(_filter)
    return pipeline

if __name__ == '__main__':
    main()
