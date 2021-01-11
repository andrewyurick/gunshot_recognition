sample_rate = 32000
clip_samples = sample_rate * 30

mel_bins = 64
fmin = 50
fmax = 14000
window_size = 1024
hop_size = 320
window = 'hann'
pad_mode = 'reflect'
center = True
device = 'cuda'
ref = 1.0
amin = 1e-10
top_db = None

#Changed to folder labels in dataset_root
labels = ['BoltAction22', 'Colt1911', 'Glock9', 'Glock45', 'HKUSP', 'Kimber45', 'Lorcin380', 
    'M16', 'MP40', 'Remington700', 'Ruger22', 'Ruger357', 'Sig9', 'Smith&Wesson22', 'Smith&Wesson38special', 'SportKing22', 'WASR-10', 'WinchesterM14']
    
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
classes_num = len(labels)
