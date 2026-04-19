# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-59b36a090be7bc29212f1b29aa7bc29b12f23e5a450aaf05d1c3eab4e44abc8c`
- fixture hash: `sha256-712180e16240a5850bc7f0f166cbbaa035f07312fa10c3e98606123034cbbf4c`
- score hash: `sha256-d6db8fccc7fa5925b6bde70007c98da7e7c1264be596b66c1cb52eac9675cefc`
- bundle hash: `sha256-9cbccb760a81245bec0cf3a3b830f2532ca6d4380a31879e7dfa84cef0057768`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | learned_route | 60 |
| 3 | vector_only | 60 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/12
- phrase hit rate: 0.25

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 1 | 1 |
| learned_route | 1 | 1 | 0.333333 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9b108d912f44bc4b15526ab7db40ee75e868952f9ef4952b3a83ae96ae65d4c1 |
| vector_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-bc8e847b224ab25f1f4d436db18a69819da0c308beb4e3354c9f40beef2c48a2 |
| graph_prior_only | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 1 | sha256-819d1af45542c79ec38281f7b34641ed5a4f0663f1a42bd6f1c4bebf81bf8327 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-c3bbaa76c05772c535d3449851869468fe0eb94c2fae2fedf0395774df0cec56 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-5e7e632e | sha256-c7f95684196210533ec3b2f0e30616ccaa2c863e4faa864f7bbf866292c5c506 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | yes | no | pack-5e7e632e | sha256-3c43c4babc605b755579bd5c7467804e28aab1f30725fa7abfa5f87de7c30687 |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-5e7e632e | sha256-c7f95684196210533ec3b2f0e30616ccaa2c863e4faa864f7bbf866292c5c506 |
