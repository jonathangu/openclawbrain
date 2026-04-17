# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-59b36a090be7bc29212f1b29aa7bc29b12f23e5a450aaf05d1c3eab4e44abc8c`
- fixture hash: `sha256-712180e16240a5850bc7f0f166cbbaa035f07312fa10c3e98606123034cbbf4c`
- score hash: `sha256-1902d9f28972f5aa118c348ac506fc46273fdeb2b91c19f86f3e60e2f51b6060`
- bundle hash: `sha256-f8b34c09ccb68bba17e562f849cf7321b65ffbed1301cdbcae8c5a368e31c309`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 60 |
| 2 | vector_only | 60 |
| 3 | learned_route | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 2/12
- phrase hit rate: 0.166667

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.333333 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.333333 | 0 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-9b108d912f44bc4b15526ab7db40ee75e868952f9ef4952b3a83ae96ae65d4c1 |
| vector_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-79dc55976d2f0d86a693a76747b20573ab27d99d1b6ed09b2bc0e3c78c84d598 |
| graph_prior_only | 1 | 1 | 1/3 | 0 | 0 | 1 | 0 | 1 | sha256-93526e3ca632769b08e32671f6344a1c2aeb7434b0c65d01bb0b44ec4428908b |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-c60bacb284670a2db0c39be9458a73129b857620988f0872ef9019fa85b9697b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 60 | yes | 1/3 | no | no | pack-192578cb | sha256-7dc23c7a29015b7b64e6a57026c45d4bbb93d99809921e4362a068559638d010 |
| graph_prior_only | turn-1 | 60 | yes | 1/3 | no | no | pack-192578cb | sha256-ef6180256d579d4ec80590ec61165edc10ba0fe2cd26a36098037f4ac4dcec17 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-b8d649d2 | sha256-67e1b9ce4d4fae8fc21b748fe6fc8d2b3cc2a8502bc25f5a978619b9f757b1dd |
