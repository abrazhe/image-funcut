#!/usr/bin/env python



import os
import argparse

import json

from functools import partial

import numpy as np
from scipy import ndimage,signal

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from imfun import ofreg, multiscale, fseq, core, ui
from imfun.ofreg import imgreg, stackreg, warps

_smoothing_filters = {'gaussian': ndimage.gaussian_filter,
                      'atrous': multiscale.atrous.smooth,
                      'atrous.detrend': multiscale.atrous.detrend,
                      'rescale':lambda f,par: core.rescale(f),
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
                               available models: {shifts, mslkp,msclg, affine, Greenberg-Kerr, homography}')),
        '-t' : ('--type', dict(default='template',
                                  choices = ['template', 'pca','recursive'],
                                  help='stabilization type')),
        '-n': ('--ncpu', dict(default=4, type=int, help="number of CPU cores to use")),
        '--record': dict(default=None, help='record within file to use (where applicable)'),
        '-v': ('--verbose', dict(action='count', help='increment verbosity level')),
        '--with-movies': dict(action='store_true'),
        '--suff': dict(default='', help="optional suffix to append to saved registration recipe"),
        '--fps': dict(default=25,type=float,help='fps of exported movie'),
        '--dct-encode': dict(action='store_true'),
        '--pca-denoise': dict(action='store_true'),
        '--bitrate':dict(default=16000,type=float, help='bitrate of exported movie'),
        '--no-zstacks': dict(action='store_true', help='try to avoid z-stacks')
        }
    for arg,kw in list(argdict.items()):
        if isinstance(kw, dict):
            parser.add_argument(arg,  **argdict[arg])
        else:
            parser.add_argument(arg, kw[0], **kw[1])

    args = parser.parse_args()

    if args.model is None:
        args.model = ['msclg']
    if args.smooth is None:
        args.smooth = []
    else:
        for m in args.model:
            if not isinstance(m, str) and len(m)>1:
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
            for key,val in list(pars.items()):
                setattr(args, key, val)
            
    if args.verbose > 2:
        print(args)

    #registrators = opflowreg.RegistrationInterfaces

    def apply_reg(frames):
        if args.type == 'template':
            if args.verbose > 1:
                print('stabilization type is template')
            tstart = int(len(frames)/2)
            tstop = min(len(frames),tstart+50)
            template = np.max(frames[tstart:tstop],axis=0)
            def register_stack(stack, registrator, **fnargs):
                return stackreg.to_template(stack,template,registrator,njobs=args.ncpu,**fnargs)
        elif args.type in ['pca', 'pca-5']:
            print('in pca')
            templates, indices = fseq.frame_exemplars_pca_som(frames,som_gridshape=(5,1))
            if args.verbose>1:
                print("Created frame exemplars by clustering PCA coefficients")
            def register_stack(stack, registrator, **fnargs):
                return stackreg.to_templates(stack, templates,indices,registrator,njobs=args.ncpu,**fnargs)
        elif args.type == 'recursive':
            def register_stack(stack, registrator,**fnargs):
                return stackreg.recursive(stack,registrator,**fnargs)[1]
        else:
            raise NameError("Unknown registration type")
        # TODO: below is just crazy. has to be made neat later
        reg_dispatcher = {'affine':imgreg.affine,
                          'homography':imgreg.homography,
                          'shifts':imgreg.shifts,
                          'Greenberg-Kerr':imgreg.greenberg_kerr,
                          'mslkp':imgreg.mslkp,
                          'msclg':imgreg.msclg}
        operations = args.model
        newframes = frames
        warp_history = []
        for movement_model in operations:
            if not isinstance(movement_model, str):
                if len(movement_model)>1:
                    model, model_params = movement_model
                else:
                    model, model_params = movement_model[0],{}
            else:
                model = movement_model
                model_params = {}
            if args.verbose > 1:
                print('correcting for {} with params: {}'.format(model, model_params))
            warps = register_stack(newframes, reg_dispatcher[model], **model_params)
            warp_history.append(warps)
            newframes = ofreg.warps.map_warps(warps, newframes, njobs=args.ncpu)
        final_warps = [ofreg.warps.compose_warps(*warpchain) for warpchain in zip(*warp_history)]
        del newframes
        return final_warps
    
    for stackname in args.imagestacks:
        out_suff = make_outname_suffix(args)

        outname = fseq._is_glob_or_names(stackname) and os.path.split(stackname)[0] or stackname
        outname += out_suff

        out_stab_name = outname + '.stab'
        #try:
        if True:
            if args.verbose:
                print('\nCalculating motion stabilization for file {}'.format(stackname))
                
            if args.pretend: continue
            #fskwargs =
            fs = fseq.from_any(stackname, ch=args.ch, record=args.record)

            if 'no_zstacks' and guess_fseq_type(fs) == 'Z':
                continue

            ## use first 20 components just as test for now
            ncomp = 20
            if args.pca_denoise:
                 print('start PCA denoising')
                 data = fs.as3darray()
                 emp_mean = data.mean(0)
                 from imfun import pica
                 data = data-emp_mean
                 u,s,vh = np.linalg.svd(pica.reshape_from_movie(data), full_matrices=False)
                 rec = u[:,:ncomp].dot(np.diag(s[:ncomp]).dot(vh[:ncomp]))
                 rec = pica.reshape_to_movie(rec, emp_mean.shape)+emp_mean
                 fs = fseq.from_array(rec)
                 print('PCA denoising done')

            smoothers = get_smoothing_pipeline(args.smooth)
            if args.verbose>1:
                print('created smoothers list')
            fs.frame_filters = smoothers[:]
            if args.verbose> 1:
                print('starting warps')
            warps = apply_reg(fs)
            if args.verbose >1:
                print('Calculated warps')
            if args.dct_encode:
                ofreg.warps.to_dct_encoded(out_stab_name, warps)
                # Saving with numpy creates smaller files than pickle, but attaches a
                # .npy extension 
                if os.path.exists(out_stab_name+'.npy') and not os.path.exists(out_stab_name):
                    out_stab_name = out_stab_name+'.npy'
                # When using DCT coding of frames, it makes sense to use the encoded frames
                # for demonstration 
                warps = ofreg.warps.from_dct_encoded(out_stab_name)
            else:
                ofreg.warps.to_pickle(out_stab_name,warps)
            if args.verbose:
                print('saved motions stab recipe to {}'.format(out_stab_name))
            del fs
            
            if args.with_movies:

                fsall = fseq.from_any(stackname, record=args.record)
                if len(fsall.stacks)>2:
                    fsall.stacks = fsall.stacks[:-1] # drop blue channel

                vl,vh = fsall.data_percentile((0.5, 99.5))    
                #vl, vh = fsall.data_percentile(0.5), fsall.data_percentile(99.5)
                #vl,vh = np.min(vl), np.max(vh)
                #if args.verbose > 10:
                #    print 'vl, vh: ', vl, vh


                fs2 = ofreg.warps.map_warps(warps, fsall, njobs=args.ncpu)

                p1 = ui.Picker(fsall)
                p2 = ui.Picker(fs2)




                #proj1 = fsall.time_project(fn=partial(np.mean, axis=0))
                #proj2 = fs2.time_project(fn=partial(np.mean, axis=0))

                fig,axs = plt.subplots(1,2,figsize=(12,5.5))
                #def _lutfn(f): return np.clip((f-vl)/(vh-vl), 0, 1)
                #def _lutfn(f): return np.dstack([np.clip(f[...,k],vl[k],vh[k])/vh[k] for k in range(f.shape[-1])])
                for ax,p,t in zip(axs,(p1,p2),['before','stabilized']):
                    p.clims = list(zip(vl,vh))
                    ax.imshow(p._lutconv(p.home_frame),aspect='equal')
                    #imh = ax.imshow(f[...,0],aspect='equal',vmin=vl,vmax=vh); plt.colorbar(imh,ax=ax)
                    plt.setp(ax, xticks=[],yticks=[],frame_on=False)
                    ax.set_title(t)
                plt.savefig(outname+'-average-projections.png')
                fig.clf()
                plt.close(fig)

                if args.verbose > 2:
                    print(outname+'-stabilized-video.mp4')

                
                ui.pickers_to_movie([p1, p2],
                              outname+'-stabilized-video.mp4',
                              titles=['before', 'stabilized'],
                                    bitrate=args.bitrate)

                del fsall, fs2
                
                plt.close('all')
            del warps
            if args.verbose>2: print('Done')
        else:
        #except Exception as e:
            print("Couldn't process {} because  {}".format(stackname, e))
        
def guess_fseq_type(fs):
    dz = fs.meta['axes'][0]
    units = str(dz.unit)
    out = ''
    if units in ['um','mm','m','nm']:
        out = 'Z'
    elif units in ['ms','us', 's', 'msec', 'usec', 'ns', 'sec']:
        out = 'T'
    return out

def make_outname_suffix(args):
    models = [isinstance(m,str) and m or m[0] for m in args.model]
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
        #conv = (name == 'median') and int or float
        conv = lambda x: int(float(x)) == float(x) and int(float(x)) or float(x)
        def _filter(f):
            return smoother(f.astype(np.float), conv(par))
        pipeline.append(_filter)
    return pipeline

if __name__ == '__main__':
    main()
