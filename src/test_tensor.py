from tensorflow.python import pywrap_tensorflow  
import os
checkpoint_path =  "./data/model_checkpoints/squeezeDet/model.ckpt-87000"
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  
var_to_shape_map = reader.get_variable_to_shape_map()  
for key in var_to_shape_map:  
    print("tensor_name: ", key)  
    if key == 'fire2/squeeze1x1/kernels/Momentum':
	print(reader.get_tensor(key).shape)
    #print(reader.get_tensor(key)) # Remove this is you want to print only variable names  

