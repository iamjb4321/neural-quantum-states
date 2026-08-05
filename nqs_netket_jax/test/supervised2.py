import h5py
import numpy as np
from tenpy.tools.hdf5_io import load, load_from_hdf5


h5_filepath = r"C:\Users\jj9ba\Research\DREU QIS-AI\AI Model\neural-quantum-states\three_site_Kagome_Hubbard_296.h5"
#try:
with h5py.File(h5_filepath, 'r+') as f_h5:
    
    #psi = load_from_hdf5(f_h5, 'psi')
    #print(help(load_from_hdf5))
    #obs_group = load_from_hdf5(f_h5, 'observables')

    print("Keys:", list(f_h5['observables'].keys()))
    for key in list(f_h5['observables'].keys()):
        data_pointer = f_h5['observables'][key]
        data = data_pointer[:]
        data_pointer2 = f_h5['average_observables'][key]
        data2 = data_pointer2[:]
        print(key, " (0,0):\n", data)
        #print(key, " AVERAGE:\n", data2)

filename='tenpy_dataset.npz'
num_epochs=1000
lr=1e-3

print("--- Step 2: Training Supervised NQS in JAX ---")
    
    # Load dataset
data = np.load(filename)
X_data = jnp.array(data['X'])
y_data = jnp.array(data['y'])
L = X_data.shape[1]
print(f"X_data: {X_data}")

