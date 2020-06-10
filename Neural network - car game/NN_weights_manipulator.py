import numpy as np
import tensorflow as tf


load_name_blank = "DDPG_17-42-45_0.10_ 83"
load_name_base = "DDPG_12-01-38_2.00_281"

for model_type in ["actor","critic"]:
	# load new model
	model_blank = tf.keras.models.load_model(load_name_blank+ "/" + model_type)
	blank_layers = [(x.numpy(),y.numpy()) for x,y in zip(model_blank.weights[0::2],model_blank.weights[1::2])]
	# print(model_blank)
	# load base model
	model_base = tf.keras.models.load_model(load_name_base+ "/" + model_type)
	base_layers = [(x.numpy(),y.numpy()) for x,y in zip(model_base.weights[0::2],model_base.weights[1::2])]
	# print(model_base)
	# calculate blank weights
	print([(i[0].shape,i[1].shape) for i in blank_layers])
	print([(i[0].shape,i[1].shape) for i in base_layers])

	new_layers = []
	for layer_no, ((blank_weights, blank_bias), (base_weights, base_bias)) in enumerate(zip(blank_layers, base_layers)):
		# print("")
		# print(layer_no)
		# print(blank_weights.shape, base_weights.shape, blank_weights.shape == base_weights.shape)
		if layer_no == 0:
			new_bias = base_bias
			# split repeated from non-repeated weights
			temp_blank_weights_dist, temp_blank_weights_misc = np.array_split(blank_weights,[blank_weights.shape[0]-1], axis=0)
			# print(temp_blank_weights_dist.shape, temp_blank_weights_misc.shape)
			temp_base_weights_dist, temp_base_weights_misc = np.array_split(base_weights,[base_weights.shape[0]-1], axis=0)
			# print(temp_base_weights_dist.shape, temp_base_weights_misc.shape)
			# split repeated weights
			mem_len = temp_blank_weights_dist.shape[0]-temp_base_weights_dist.shape[0]
			# print(mem_len)
			temp_blank_weights_dist_split = np.array_split(temp_blank_weights_dist,mem_len, axis=0)
			# print([i.shape for i in temp_blank_weights_dist_split])
			temp_base_weights_dist_split = np.array_split(temp_base_weights_dist,mem_len, axis=0)
			# print([i.shape for i in temp_base_weights_dist_split])
			# copy weights
			temp_new_weights_dist_split = []
			for blank, base in zip(temp_blank_weights_dist_split, temp_base_weights_dist_split):
				new = blank
				new[:base.shape[0],:]=base
				temp_new_weights_dist_split.append(new)
			# print([i.shape for i in temp_new_weights_dist_split])
			# join repeated new weights
			temp_new_weights_dist = np.concatenate(temp_new_weights_dist_split)
			# print(temp_new_weights_dist.shape)
			new_weights = np.concatenate((temp_new_weights_dist, temp_base_weights_misc))
		elif layer_no == 2:
			new_weights = blank_weights
			new_weights[:,:base_weights.shape[1]]=base_weights
			new_bias = blank_bias
			new_bias[:base_bias.shape[0]]=base_bias
		else:
			new_bias = base_bias
			new_weights = base_weights
		new_layers.append((new_weights, new_bias))
	print([(i[0].shape,i[1].shape) for i in new_layers])

	# apply new weights
	model_new = model_blank
	model_new_weights=[]
	for (weights, bias) in new_layers:
		model_new_weights.append(weights)
		model_new_weights.append(bias)
	print([i.shape for i in model_new_weights])
	for i in range(len(model_new.weights)):
	    model_new.weights[i].assign(model_new_weights[i])
	model_new.save(load_name_base+ "_NEW/" + model_type)