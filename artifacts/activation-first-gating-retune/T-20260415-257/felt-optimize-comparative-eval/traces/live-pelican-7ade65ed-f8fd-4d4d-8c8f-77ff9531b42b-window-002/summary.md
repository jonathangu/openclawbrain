# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-7ade65ed-f8fd-4d4d-8c8f-77ff9531b42b-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-1cc116d5a5a3e4268eee5081d6d597a83a2afaebb6c2529b01952ad2f45437c1`
- fixture hash: `sha256-ccd8a0f1240cc7f92941ab2c1ede0327e4ed0a420f6a51ec4c81e0437c7d59e2`
- score hash: `sha256-88ae36f8905d878fc0022e9804662544ff6fa07455e398683080fdb840fccb84`
- bundle hash: `sha256-4317a947da82f01333535d7319c551063dab3d0dc067254aaf59e05e0ed44789`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a97ee439ef356b4483f5735f34054ec24021480ea2dadec6ac22262eafbebd17 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-999cbf0dbbd33e04890488375d58a108ee0a80cec04ac0685c596f9efe9cf453 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-cb955d0c9e6c4d85ee6de269cedb0dcdfd87e9ee1fe5c0ddbacf89c165afacae |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-e795a5c4317bd259369992fc73dcb29ceada79ed7b0f00c4cef4aaa7d7d4c374 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7e7a4b77 | sha256-de424d29fd5bd41c91830523bb8d4c8a365e34e240c0019f1d1a236ca7b0c488 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-7e7a4b77 | sha256-3478acb9e497b4bed837c0fcd5fdf6a06ffa334ff2df37bdec604913e2cfa7dd |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-86e245fe | sha256-bba3348a3198d0580253d5b5b3cf7f4e9659bc82e1779c469efddd029a67ba7d |
