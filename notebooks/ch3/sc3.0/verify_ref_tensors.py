#!/usr/bin/env python
"""
sc3.0 -- validation-ladder step 1 (design doc SS7): every PGDr-PIKAN
reference-element tensor (mass, stiffness, convection, triple products,
subdivision/transfer matrices, endpoint values -- doc SS1) checked against
an independent Gauss-Legendre quadrature implementation. No training, no
network -- pure linear algebra, expected to pass in seconds.

Also reprints the doc's own hand-worked check values (SS1.3's
K11=2,K22=6,K33=12,K44=20,K13=2,K24=6,K35=12 and SS1.5's T^L up to degree 4)
so the numbers in this script's output can be read directly against the
design doc.
"""
import json
import os
import sys
import time

ROOT = '/home/df.gomezb/Tesis'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from pinns_framework.pgdr import (
    stiffness_matrix,
    transfer_matrices,
    verify_all,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print('=== sc3.0 | reference-tensor verification (doc SS1, SS7 step 1) ===')

    K = stiffness_matrix(6)
    checks = {
        'K11': K[1, 1].item(), 'K22': K[2, 2].item(), 'K33': K[3, 3].item(),
        'K44': K[4, 4].item(), 'K13': K[1, 3].item(), 'K24': K[2, 4].item(),
        'K35': K[3, 5].item(),
    }
    expected = {'K11': 2, 'K22': 6, 'K33': 12, 'K44': 20, 'K13': 2, 'K24': 6, 'K35': 12}
    print('stiffness check values (doc SS1.3):')
    all_match = True
    for name, val in checks.items():
        ok = abs(val - expected[name]) < 1e-12
        all_match &= ok
        print(f'  {name} = {val:.1f}  (expected {expected[name]}, {"OK" if ok else "MISMATCH"})')

    TL = transfer_matrices(5, side=-1)
    print('\nT^L up to degree 4 (doc SS1.5):')
    torch.set_printoptions(precision=6, sci_mode=False)
    print(TL)

    print('\nrunning verify_all() -- cross-checks M, K, C, T, D, T^L, T^R '
          'against independent Gauss-Legendre quadrature ...')
    t0 = time.perf_counter()
    verify_all(N=20, N_expensive=12)
    wall = time.perf_counter() - t0
    print(f'verify_all() PASSED in {wall:.2f}s')

    summary = {
        'stiffness_checks': checks,
        'stiffness_checks_expected': expected,
        'stiffness_checks_match': all_match,
        'verify_all_passed': True,
        'verify_all_wall_s': wall,
    }
    with open(os.path.join(RESULTS_DIR, 'ref_tensors_report.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nreport written to {os.path.join(RESULTS_DIR, 'ref_tensors_report.json')}")


if __name__ == '__main__':
    main()
