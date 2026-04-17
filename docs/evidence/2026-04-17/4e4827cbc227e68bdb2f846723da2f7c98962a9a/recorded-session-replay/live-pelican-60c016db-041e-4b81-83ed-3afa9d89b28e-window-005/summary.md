# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-60c016db-041e-4b81-83ed-3afa9d89b28e-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f4905564adc9cb953b8b5504309a4080c3ac583fe0f629cb62b1e05f91ea23a3`
- fixture hash: `sha256-0ad0b5e1e0f2271069ee0d118e38a8f083b22de4d11f9b10cb9ee63b3ed54883`
- score hash: `sha256-807240f8a599384b41e08fcea1500a0df9bc48e80fbc85b750fde3fb229ad0c3`
- bundle hash: `sha256-76a254072f8fb4165f78685439bbd2d0342e7cc957adcfb1b500d71c7ee3ca13`

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
- phrase hits: 0/8
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-64badee520388e2e251dcf80ba87d74776085beb63219f4be30791f06cfae40c |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-a5733003af75997345b801f9a1b9d3288f7072107eb9d0019d18efb7af3d0404 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-77ae0827bf1bda039056440cff95d3c0e52a1d5b47528393778bfb0ef9cc8ae5 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-cb63875f3e4b06b7573008d20dfc6f1ee740bda4b6162dc081aed55306dba692 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d6abeadb | sha256-82f7a2bea58b3bb0637616cf0526a8653456d5c2880000178891fff670337618 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-d6abeadb | sha256-7e1d08052f5439f1c65b91cad1e3d6bb2c4f4a5612cda8f2a3e0e87e55b43af3 |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-9e284468 | sha256-c7e96570c47320e3a0c64076940f699432f0896174e1a7b41d3248842db20450 |
