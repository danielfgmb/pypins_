#!/usr/bin/env python
"""
sc3.0 -- validation-ladder step 2 (design doc SS7, SS3.1): confirm that a
composed multi-layer Legendre-KAN is *exactly* a polynomial of degree
p^n_layers -- the whole argument for using Legendre edges instead of
pyKAN's B-splines (composed B-splines are not splines on the same knots, so
this closure would not hold for them).

Test: evaluate a random-weight multi-layer LegendreKAN on a fine grid,
project it onto the Legendre basis at degree p^n_layers via Gauss-Legendre
quadrature (kan.project_to_legendre), reconstruct, and check the residual
against the raw network output is machine-zero. Swept over per-edge degree
p in {2..6} and over 1/2/3 layers (the 1-layer case is the trivial
Bubnov-Galerkin case, degree p, c == theta) -- if closure were only
approximate, the residual would grow with p^L; instead it stays pinned at
float round-off across the whole sweep, which is the point (SS3.1: exact
because polynomials are closed under composition/products, not merely
"small enough to not matter").
"""
import json
import os
import sys

ROOT = '/home/df.gomezb/Tesis'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from pinns_framework.pgdr import LegendreKAN, closure_residual

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DOMAIN = (-1.0, 1.0)
DEGREES = [2, 3, 4, 5, 6]
LAYER_CONFIGS = [
    ('1-layer (Bubnov-Galerkin case, phase 1)', (1, 1)),
    ('2-layer', (1, 2, 1)),
    ('3-layer', (1, 2, 2, 1)),
]


def main():
    print('=== sc3.0 | KAN closure test (doc SS3.1, SS7 step 2) ===')
    torch.manual_seed(0)

    report = []
    for label, widths in LAYER_CONFIGS:
        n_layers = len(widths) - 1
        for degree in DEGREES:
            net = LegendreKAN(widths=widths, degree=degree, domain=DOMAIN)
            expected_degree = degree ** n_layers
            res = closure_residual(net, DOMAIN)
            passed = res < 1e-9
            print(f'{label:45s} p={degree} p^L={expected_degree:4d}  '
                  f'residual={res:.3e}  {"PASS" if passed else "FAIL"}')
            report.append({
                'label': label, 'widths': list(widths), 'n_layers': n_layers,
                'degree': degree, 'p_pow_L': expected_degree,
                'residual': res, 'passed': passed,
            })
        print()

    all_passed = all(r['passed'] for r in report)
    with open(os.path.join(RESULTS_DIR, 'kan_closure_report.json'), 'w') as f:
        json.dump({'domain': list(DOMAIN), 'degrees': DEGREES,
                    'cases': report, 'all_passed': all_passed}, f, indent=2)
    print(f"{'ALL PASSED' if all_passed else 'SOME FAILED'} -- report written to "
          f"{os.path.join(RESULTS_DIR, 'kan_closure_report.json')}")


if __name__ == '__main__':
    main()
