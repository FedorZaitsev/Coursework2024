import torch
import os
from .img_clf_train_pipeline import clf_train, evaluate


def static_quantize(model, loader, loss_fn):
    # Our initial baseline model which is FP32
    model_fp32 = model
    model_fp32.eval()

    # Sets the backend for x86
    model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')

    # Prepares the model for the next step i.e. calibration.
    # Inserts observers in the model that will observe the activation tensors during calibration
    model_fp32_prepared = torch.quantization.prepare(model_fp32, inplace = False).to('cpu')

    model_fp32_prepared.eval()
    # Calibrate over the train dataset. This determines the quantization params for activation.
    evaluate(model_fp32_prepared, loader, None, loss_fn)

    # Converts the model to a quantized model(int8)
    model_int8 = torch.quantization.convert(model_fp32_prepared) # Quantize the model

    def print_size_of_model(model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p")/1e6)
        os.remove('temp.p')

    print('Initial model size:')
    print_size_of_model(model_fp32)
    print('Quantized model size:')
    print_size_of_model(model_int8)

    return model_int8
