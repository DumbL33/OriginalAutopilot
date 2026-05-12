import gc
    
del go_straight_engine
del go_right_engine
del go_left_engine
del lstm_engine_park_short
del lstm_engine_park_long
del lstm_engine_park_out
del ddrnet_engine
    
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
