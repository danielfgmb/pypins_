#!/usr/bin/env python
"""
sc3.0 -- validation-ladder step 2 (design doc SS7, SS3.1): confirm that a
composed multi-layer Legendre-KAN is *exactly* a polynomial of degree
p^n_layers -- the whole argument for using Legendre edges instead of
pyKAN's B-splines (composed B-splines are not splines on the same knots, so
this closure would not hold for them).

Test: evaluate a random-weight 2-layer LegendreKAN on a fine grid, project
it onto the Legendre basis at degree p^2 via Gauss-Legendre quadrature
(kan.project_to_legendre), reconstruct, and check the residual against the
raw network output is machine-zero. Also checks a 3-layer network (degree
p^3) and the trivial 1-layer case (degree p, c == theta) for completeness.
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
DEGREE = 4


def main():
    print('=== sc3.0 | KAN closure test (doc SS3.1, SS7 step 2) ===')
    torch.manual_seed(0)

    configs = [
        ('1-layer (Bubnov-Galerkin case, phase 1)', (1, 1)),
        ('2-layer', (1, 2, 1)),
        ('3-layer', (1, 2, 2, 1)),
    ]

    report = []
    for label, widths in configs:
        net = LegendreKAN(widths=widths, degree=DEGREE, domain=DOMAIN)
        expected_degree = DEGREE ** (len(widths) - 1)
        res = closure_residual(net, DOMAIN)
        passed = res < 1e-9
        print(f'{label:45s} widths={widths} p^L={expected_degree:4d}  '
              f'residual={res:.3e}  {"PASS" if passed else "FAIL"}')
        report.append({
            'label': label, 'widths': list(widths), 'degree': DEGREE,
            'p_pow_L': expected_degree, 'residual': res, 'passed': passed,
        })

    all_passed = all(r['passed'] for r in report)
    with open(os.path.join(RESULTS_DIR, 'kan_closure_report.json'), 'w') as f:
        json.dump({'degree': DEGREE, 'domain': list(DOMAIN),
                    'cases': report, 'all_passed': all_passed}, f, indent=2)
    print(f"\n{'ALL PASSED' if all_passed else 'SOME FAILED'} -- report written to "
          f"{os.path.join(RESULTS_DIR, 'kan_closure_report.json')}")


if __name__ == '__main__':
    main()
