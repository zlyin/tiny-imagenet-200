=======================================
- Experiements with DeeperGoogLeNet
=======================================

# Experiment #1
- learning rates
| Epoch | Learning Rate |
|-------|---------------|
| 1->25 | 1e-2			|
| 26->35 | 1e-3			|
| 36->95 | 1e-4			|
- without `4a -> 4e` inception_modules & the following MaxPooling2D
- Use `SGD` and `momentum=0.9`, no nesterov momentum term yet
- Test accuracy over stages:
	- @Stage 1:	
		```
		[INFO] loading ./output/exp1/checkpoints/epoch_25.hdf5 ...
		[INFO] rank-1 acc = 35.270%
		[INFO] rank-5 acc = 61.400%
		```
	- @Stage 2:
		```
		[INFO] loading ./output/exp1/checkpoints/epoch_35.hdf5 ...
		[INFO] rank-1 acc = 39.370%
		[INFO] rank-5 acc = 65.650%
		```
	- @Stage 3:
		```
		[INFO] loading ./output/exp1/checkpoints/epoch_95.hdf5 ...
		[INFO] rank-1 acc = 41.050%
		[INFO] rank-5 acc = 66.900%

		[INFO] loading ./output/exp1/checkpoints/epoch_85.hdf5 ...
		[INFO] rank-1 acc = 41.000%
		[INFO] rank-5 acc = 66.790%

		[INFO] loading ./output/exp1/checkpoints/epoch_75.hdf5 ...
		[INFO] rank-1 acc = 41.290%
		[INFO] rank-5 acc = 66.940%
		```
- From learning curve, training has stagnated; 
- After reducing learning rate to 1e-4, train loss quickly plateaued, though val
	loss still oscillates. This states that either NN is not powerful enough or
	SGD can't reach a lower loss;

# Experiment #2
- learning rates
| Epoch | Learning Rate |
|-------|---------------|
| 1->25 | 1e-3			|
| 26->35 | 1e-4			|
| 36->60 | 1e-5			|
- Try switching out `SGD` for `Adam` optmizer before changing NN structure;
- Test accuracy over stages:
	- @Stage 1:	
		```
		[INFO] loading ./output/exp2/checkpoints/epoch_25.hdf5 ...
		[INFO] rank-1 acc = 33.800%
		[INFO] rank-5 acc = 61.370%
		```
	- @Stage 2:	
		```
		[INFO] loading ./output/exp2/checkpoints/epoch_35.hdf5 ...
		[INFO] rank-1 acc = 41.900%
		[INFO] rank-5 acc = 67.930%
		```
	- @Stage 3:	
		```
		[INFO] loading ./output/exp2/checkpoints/epoch_60.hdf5 ...
		[INFO] rank-1 acc = 42.250%
		[INFO] rank-5 acc = 68.080%
		```

- from learning curve, loss droped quickly after reducing lr at Epoch 25;
- training has stagnated after Epoch 40; but the final rank1-acc is 1% higher
	than that of Epx1; therefore, `Adam` is still useful!
- we need a more deeper NN to learn more discriminative patterns;

# Experiment #3

| Epoch | Learning Rate |
|-------|---------------|
| 1->40 | 1e-3			|
| 41->60 | 1e-4			|
| 61->80 | 1e-5			|
- add back without `4a -> 4e` inception_modules & the following MaxPooling2D;
- still use `Adam` optimizer;  
- Test accuracy over stages:
	- @Stage 1:
		```
		[INFO] loading ./output/exp3/checkpoints/epoch_40.hdf5 ...
		[INFO] rank-1 acc = 31.780%
		[INFO] rank-5 acc = 56.150%
		```
	- @Stage 2:
		```
		[INFO] loading ./output/exp3/checkpoints/epoch_60.hdf5 ...
		[INFO] rank-1 acc = 44.230%
		[INFO] rank-5 acc = 70.400%
		```
	- @Stage 3:
		```
		[INFO] loading ./output/exp3/checkpoints/epoch_80.hdf5 ...
		[INFO] rank-1 acc = 45.860%
		[INFO] rank-5 acc = 71.620%
		```
- reduce learning rate twice, loss drop immediately; add back `4a -> 4e`
	incepttion_module boosts 3% acc!
- refine conv_module by adjusting the order of `Conv => BN => Act` to `Conv => Act => BN`;

# Experiment #4

| Epoch | Learning Rate |
|-------|---------------|
| 1->40 | 1e-3			|
| 41->80 | 1e-4			|
| 81->100 | 1e-5		|
- refine conv_module by adjusting the order of `Conv => BN => Act` to `Conv => Act => BN`;
- Test accuracy over stages:
	- @Stage 1:
		```
		[INFO] loading ./output/exp4/checkpoints/epoch_40.hdf5 ...
		[INFO] rank-1 acc = 37.380%
		[INFO] rank-5 acc = 63.810%
		```
	- @Stage 2:
		```
		INFO] loading ./output/exp4/checkpoints/epoch_80.hdf5 ...
		[INFO] rank-1 acc = 46.320%
		[INFO] rank-5 acc = 71.080%
		```
	- @Stage 3:
		```
		[INFO] loading ./output/exp4/checkpoints/epoch_100.hdf5 ...
		[INFO] rank-1 acc = 47.440%
		[INFO] rank-5 acc = 72.140%
	
		```
- easily to observe that loss drops immediately after reducing learning rates;
- loss plateaus and overfitting appears;
- use the order of `Conv => Act => BN` is able to boost acc up by 1.5%


=======================================
- Experiments with ResNet
=======================================
# Experiment #4

| Epoch | Learning Rate |
|-------|---------------|
| 1->25 | 1e-1			|
| 25->40| 1e-2			|
| 41->70| 1e-3			|
| 71->90| 1e-4			|
- use 2GPUs to achieve distributed training;
- Add `Conv => Act => BN => ZeroPadding => MaxPool` to the fist block of ResNet
	- `ResNet.build(64, 64, 3, cfg.NUM_CLASSES, [3, 4, 6], [64, 128, 256, 512], 
                reg=5e-4, bnEps=2e-5, bnMom=0.9, dataset="tiny-imagenet")`
- Test accuracy over stages:
	- @Stage 1:
		```
		[INFO] loading output/resnet_exps/exp1/epoch_25.hdf5 ...
		[INFO] rank-1 acc = 13.390%
		[INFO] rank-5 acc = 34.030%
		```
	- @Stage 2:
	- @Stage 3:


To record & test!


