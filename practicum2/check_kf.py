import numpy as np
from hashlib import sha1
from warnings import warn

def get_hs(x):
    return sha1(x).hexdigest()

def check_kf_init(kf):
    """
     DON'T LOOK IN HERE, THERE IS NOTHING TO SEE HERE!
     (don't mind the man behind the curtain)
     The true matrices are not in here.
    
     Instead, the sanity check applies a hash to your
     matrices and compares it to the expected outcome
     of the correct solutions.
    """
    is_ok = True
    hs = {
            'F': 'c9b37406a4075fb06c4216cc2c9720a2239287c2',
            'H': '34059074aa83eea6640989cc53d30fc7a4e672ce',
            'Sigma_x': '7e95b2f50d7c0b141667b5d65092f54b76619607',
            'Sigma_z': '6432f11211693a9c16ac1753731bc589b0e502c8'
            }
    # do a few sanity checks
    print('-- KF_INIT CHECKING MATRICES --\n'); 
    if kf.F.shape != (4,4): 
        is_ok = False
        warn("Size of F not as expected", RuntimeWarning)
    if kf.H.shape != (2,4): 
        is_ok = False
        warn("Size of H not as expected", RuntimeWarning)
    if kf.Sigma_x.shape != (4,4): 
        is_ok = False
        warn("Size of Sigma_x not as expected", RuntimeWarning)
    if kf.Sigma_z.shape != (2,2): 
        is_ok = False
        warn("Size of Sigma_z not as expected", RuntimeWarning)

    if get_hs(kf.F) != hs['F']: 
        is_ok = False
        warn("Content of F not as expected", RuntimeWarning)
    if get_hs(kf.H) != hs['H']: 
        is_ok = False
        warn("Content of H not as expected", RuntimeWarning)
    if get_hs(kf.Sigma_x) != hs['Sigma_x']: 
        is_ok = False
        warn("Content of Sigma_x not as expected", RuntimeWarning)
    if get_hs(kf.Sigma_z) != hs['Sigma_z']: 
        is_ok = False
        warn("Content of Sigma_z not as expected", RuntimeWarning)

    if not is_ok:
        print("Oops, something went wrong in kf.__init__!")
        is_ok = False
    if is_ok:
        print("Matrices look good. Noice!")

    print("\n")
    return is_ok

def check_kf_predict(kf):
    is_ok = True

    hs = {'mu': 'de8a847bff8c343d69b853a215e6ee775ef2ef96',
          'Sigma': '6fc6a371075b5efc7d8601a03964fcb4edbfcfcb'}

    if get_hs(kf.mu_preds[5]) != hs['mu']:
        is_ok = False
        warn("Content of mu not as expected", RuntimeWarning)
    if get_hs(kf.Sigma_preds[5]) != hs['Sigma']:
        is_ok = False
        warn("Content of Sigma not as expected", RuntimeWarning)

    if not is_ok:
        print("Oops, something went wrong in kf.predict!")
    if is_ok:
        print("Matrices look good. Noice!")
    return is_ok