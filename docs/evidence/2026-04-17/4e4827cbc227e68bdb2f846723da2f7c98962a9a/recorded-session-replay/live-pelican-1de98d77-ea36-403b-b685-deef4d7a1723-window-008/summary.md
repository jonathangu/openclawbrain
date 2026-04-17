# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-59b36a090be7bc29212f1b29aa7bc29b12f23e5a450aaf05d1c3eab4e44abc8c`
- fixture hash: `sha256-712180e16240a5850bc7f0f166cbbaa035f07312fa10c3e98606123034cbbf4c`
- score hash: `sha256-4f109a9f5ee219347c3eafb9c10bef3e308487ff81ff1b5635c5f6cb2b3caea6`
- bundle hash: `sha256-d3b931dbb60b1e64f67b87aaeaf40c68e4bf2f7ab990a064f302d29614407fce`

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
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-4b4a612f79ca477681d9f6de14a6dcacd269d769057c7df9e50ded805b63e791 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-0ae64aadffb843b9145e11f8c758ed12972195764d575473deb46e97f75936b7 |
| learned_route | 1 | 1 | 1/3 | 1 | 0 | 1 | 0 | 2 | sha256-26cbd111334dca7e55bc55dc746838d3a4aff9e0cb6f50f47e967ad05eae46f2 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-e710be20 | sha256-06b897214c6bab1d7849a3298a069b8c02a54ebc20fef7097d5d5e29b94ac97f |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-e710be20 | sha256-7fb30a956cf57c2c93b99486ca6e07741ad5ad0500cf74739d023fce652d1aab |
| learned_route | turn-1 | 60 | yes | 1/3 | yes | no | pack-86c18f27 | sha256-4e3574ee4205834adc7d7602637dfe9f9af9a21ef418af961c3265928f01b15e |
