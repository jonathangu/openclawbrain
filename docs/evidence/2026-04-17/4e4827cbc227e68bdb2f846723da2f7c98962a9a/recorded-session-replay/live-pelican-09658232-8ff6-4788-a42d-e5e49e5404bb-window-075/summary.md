# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-075`
- winner mode: `graph_prior_only`
- trace hash: `sha256-31f15910a7f37f6942dfc1fa59eebabc12b733c2fdbc101bb92672de7f721f0c`
- fixture hash: `sha256-d3d3b7c9daea7f5dceb8bcbc7d0b182082662e4eea5368602c8cfc65a5234e7a`
- score hash: `sha256-19e9732e6fb7b7395d6ed995cae10fb9ab09d5d6ad5b2575ae36287c0f692a34`
- bundle hash: `sha256-d0d177b51a7c8ecc24deb06bcede6b767cb51002132c19e7c285e19f0bfe9017`

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
- phrase hits: 0/12
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-eee7751ada66c814393120538ba88242a0ad04eb627a4b24f36524aa1be2a704 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7d264fc5c47d7d5677dd774d737685cb60a41a77e08ffeb73f13280ad6f50561 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-2a3ec35e23479a38e6f486f2036451be24274b4e215560aada3ac6dc3d9859ea |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-7df69a026337ffeddbee8ba728041d5da1f5d4608159d5a21ead248b5db88c7e |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0a7c26b8 | sha256-6e7026896b2b36a60d47507b926ca0845c1511a3854006c5a018d5c59ef507ef |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-0a7c26b8 | sha256-179ded6aa784696dd8c30ad86e39a39bf3773e85426208406db268d9d6f0b3a6 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-23638cd1 | sha256-1ca3cab903223278f1d065873b562e7db0e81be3341f8b73e33ba4aef6a4954a |
