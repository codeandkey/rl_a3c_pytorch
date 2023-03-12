#!/bin/python

"""
    factorial.py : Factorial experiment argument generator

    This file is a part of fedrl.

    USAGE:
        python factorial.py <INDEX> PARAM1=P1V1,P1V2,... PARAM2=P2V1,P2V2,...

    where INDEX is the 0-based index of the experiment to be run, and following arguments
    describe potential experiment parameters.

    EXAMPLE:
        $ python factorial.py 0 a=1,2 b=1,2
        --a=1 --b=1 --name=a=1-b=1

        $ python factorial.py 1 a=1,2 b=1,2
        --a=1 --b=2 --name=a=1-b=2

        $ python factorial.py 2 a=1,2 b=1,2
        --a=2 --b=1 --name=a=2-b=1

        $ python factorial.py 3 a=1,2 b=1,2
        --a=2 --b=2 --name=a=2-b=2

    The outputs from this command can be used as arguments to a3c.py to run an experiment over
    different combinations of parameters.
"""

import itertools
import sys

factors = []
started = False

index = int(sys.argv[1])

for arg in sys.argv[2:]:
    factors.append(dict(
        name=arg.split('=')[0],
        values = arg.split('=')[1].split(',')
    ))

experiment_values = list(itertools.product(*[f['values'] for f in factors]))
asn = []

for factor, value in zip(factors, experiment_values[index]):
    asn.append(f'{factor["name"]}={value}')
    print(f' --{asn[-1]}', end='')

print(f' --name={"-".join(asn)}')
