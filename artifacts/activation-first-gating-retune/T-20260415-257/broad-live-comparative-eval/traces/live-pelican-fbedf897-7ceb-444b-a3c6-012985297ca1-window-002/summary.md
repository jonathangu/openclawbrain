# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-fbedf897-7ceb-444b-a3c6-012985297ca1-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-4e6bcca0889e112786ccd30d6dc08d693afeab2955ee3e21db9f09dfe3094e0f`
- fixture hash: `sha256-14ff100a8ccae36fc1c57494dcba2b6e1338cfd708e5c890121212b4f7b539d1`
- score hash: `sha256-9966ff062c35fe7df2ea22f3bb292a5d8ec02595a345ef7872c1a153e84295a0`
- bundle hash: `sha256-1ebf73384a547610e40a2ad78d20a75b5d1e71f152fad858d4fc042a58119ae5`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-86a96bb1f89c4f625498603269cc86fe2157c50e9372e11582b94c39873a6510 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-59bb440842b4b69cf051362c2822ec2a34543ec4dcde9c27e5657ba68c729ab6 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-94916d5b10609b25051b49d86e5c3aaae7df72ed53f021b2a4bd07ddae11963f |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-a55ce37789cb391d18a19a584da2114e1f55375056a0114bd7906da890a9b313 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-024a22a2 | sha256-f95b36e175960d7cabc8e5a0e5533c20c84a79e95211f2f907d452fad1f40c0b |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-024a22a2 | sha256-a75d62322ada9de1a9db58c6dfdb6d4c9428de0976dcc2934486d086aaf74a34 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-8e767817 | sha256-f186b1a02247291c7e0a6b69768db3dd97fa7879d270c1a14aaf0978e70dd343 |
