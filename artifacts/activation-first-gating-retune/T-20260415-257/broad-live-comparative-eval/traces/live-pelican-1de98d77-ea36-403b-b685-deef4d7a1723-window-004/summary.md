# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-1de98d77-ea36-403b-b685-deef4d7a1723-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-570958465e5589f279bb28e850af48f0de0e358122b512d402db3214c7541c3c`
- fixture hash: `sha256-06808b26154de9486de3e390d83e02d5c54e1e0ca160f5f4c88501af04825dc3`
- score hash: `sha256-8c43a3ab780231d55bd5a9659e91aed679deb80e60814329c4dc21169e6a6a21`
- bundle hash: `sha256-baa001df8ebfd772485742321caa2a648a1ee71c2b32c3b9cb65784518af63a5`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-dd27477bdd5733b8ef83edfac9b06aafa0bfaf3753550669b2a8358e4c2d729f |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-34bf96ab41e4d6fb935f4d26895454f7a594e08af8385808cde0c3443342f1f4 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-032e0d83ca5b5d35e33084c9c0d1428c859e1c489b4319894d9b74ab1a87f20c |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-e2ae08e0e469e04d8c15f92c775291f5b15421422ce3e30f7e57666a5da2880c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-791aac92 | sha256-7bcfe465fe1f4e5913426b1085408e8b99d1d77ac90cf484fb0c34fcbc4dae5f |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-791aac92 | sha256-0a4821bd947d9630ded4d40b661ba58527f02a494f4eb292bd820dce7e409106 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-9234aefb | sha256-4a634d000285fc97bf2dcec8c48b3217de702c4297d2045e6c65c1d0fd6d6fba |
