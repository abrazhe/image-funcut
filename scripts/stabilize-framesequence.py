#!/usr/bin/env python

import argparse

import numpy as np
from scipy import ndimage,signal

from imfun import opflowreg, atrous, fseq, lib

_smoothing_filters = {'gaussian': ndimage.gaussian_filter,
                      'atrous': atrous.smooth,
                      'median': signal.medfilt}




def main():
    parser = argparse.ArgumentParser(description="""Motion stabilize image stack (frame sequence)""")
    argdict =  {
        'imagestacks' : dict(nargs='+'),
        '-ch': dict(default='r', choices='rgb012', help='color channel to use'),
        '-s': ('--smooth', dict(action='append', metavar = ('FILTER', 'PARAMETER'), nargs=2)),
        '-m': ('--model', dict(action='append',
                               choices = ['translations', 'softmesh', 'affine', 'Greenberg-Kerr'],
                               help='add movement model to use for stabilization')),
        '-v' : ('--variant', dict(default='template',
                                  choices = ['template', 'recursive'],
                                  help='stabilization variant')),
        '-n': ('--ncpu', dict(default=4, type=int, help="number of CPU cores to use")),
        '--record': dict(default=None, help='record within file to use (where applicable'),
        #'--fps': dict(default=25,type=float,help='fps of exported movie'),
        }
    for arg,kw in argdict.items():
        if isinstance(kw, dict):
            parser.add_argument(arg,  **argdict[arg])
        else:
            parser.add_argument(arg, kw[0], **kw[1])
    args = parser.parse_args()

    if args.model is None: args.model = ['softmesh']
    

    print args
    registrators = opflowreg.RegistrationInterfaces

    def apply_reg(frames, outname):
        if args.variant == 'template':
            tstart = len(frames)/2
            tstop = min(len(frames),tstart+50)
            template = np.max(frames[tstart:tstop],axis=0)
            def register_stack(stack, registrator):
                return opflowreg.register_stack_to_template(stack,template,registrator, njobs=args.ncpu)
        elif args.variant == 'recursive':
            def register_stack(stack, registrator):
                return opflowreg.register_stack_recursive(stack,registrator)[1]
        else:
            raise NameError("Unknown registration type")
        # TODO: below is just crazy. has to be made neat later
        reg_dispatcher = {'affine':registrators.affine,
                          'homograhy':registrators.homography,
                          'translations':registrators.translations,
                          'Greenberg-Kerr':registrators.greenberg_kerr,
                          'softmesh':registrators.softmesh}
        operations = args.model
        newframes = frames
        warp_history = []
        for movement_model in operations:
            warps = register_stack(newframes, reg_dispatcher[movement_model])
            warp_history.append(warps)
            newframes = opflowreg.apply_warps(warps, newframes, njobs=args.ncpu)
        final_warps = [lib.flcompose(*warpchain) for warpchain in zip(*warp_history)]
        if outname:
            opflowreg.save_recipe(final_warps, outname)
            print 'saved motions stab recipe to {}'.format()
        return newframes
    
    for stackname in args.imagestacks:
        outname = stackname + make_outname_suffix(args)
        fs = fseq.open_seq(stackname, ch=args.ch, record=args.record)
        smoothers = get_smoothing_pipeline(args.smooth)
        fs.fns.extend(smoothers)
        print 'Starting motion stabilization for file {}, saving recipy to {}'.format(stackname, outname)
        apply_reg(fs, outname)
        

def make_outname_suffix(args):
    models = args.model
    ch = "ch_{}".format(args.ch)
    if args.smooth is not None:
        smoothers = ["{}_{}".format(name,par) for name,par in args.smooth]
    else:
        smoothers = []
    return ''.join(('-','-'.join([ch]+smoothers+[args.variant]+models),'.stab'))

def get_smoothing_pipeline(smooth_entry):
    if smooth_entry is None: return []
    pipeline = []
    for name, par in smooth_entry:
        if name not in _smoothing_filters:
            raise NameError('unknown blur filter name')
        def _filter(f): return _smoothing_filters[name](f, float(par))
        pipeline.append(_filter)
    return pipeline

if __name__ == '__main__':
    main()
