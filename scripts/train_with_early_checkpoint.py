from transformers import TrainerCallback
from llamafactory.train.tuner import run_exp
class EarlyCheckpoint(TrainerCallback):
    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step==100: control.should_save=True
        return control
if __name__=='__main__': run_exp(callbacks=[EarlyCheckpoint()])
