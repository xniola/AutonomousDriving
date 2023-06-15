from tensorflow.python.compiler.tensorrt import trt_convert as trt

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=(1<<30))
conversion_params = conversion_params._replace(precision_mode="FP16")
conversion_params = conversion_params._replace(maximum_cached_engines=10)


converter = trt.TrtGraphConverterV2(input_saved_model_dir="my_model/saved_model")
converter.convert()
converter.save("tensor_rt")
