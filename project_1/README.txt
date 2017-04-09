python3 main.py [directory of images] [shutter speed file] [image format] [output name] [-r range of maximum shift in MTB (default 20)]
				[-l lambda value(default 250)] [-m method of Tonemapping (photo, durand(default))] 
				[-g gsolver(Debevec, Robertson(default))] [-v show response curve & radiance map(default False)]

You can use Execute.sh to reproduce our result (solving g with Robertson, Tonemapping with bilateral filtering)
