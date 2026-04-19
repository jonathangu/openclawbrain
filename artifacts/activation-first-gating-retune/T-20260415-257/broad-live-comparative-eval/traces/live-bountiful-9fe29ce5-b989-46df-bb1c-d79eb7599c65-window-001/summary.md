# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-9fe29ce5-b989-46df-bb1c-d79eb7599c65-window-001`
- winner mode: `graph_prior_only`
- trace hash: `sha256-3249aba74ff3b68a3a52303cdd5411f6f55111b4c4f3feb276bc9f491c4a0dfc`
- fixture hash: `sha256-ae9594a971d6ccf182aa1cfc577566bae527c792a4eca57afc1a5a898e741bd0`
- score hash: `sha256-7737ba4901415f48078d46751ea16ee31aff5884f5bcfe1d832660e961462162`
- bundle hash: `sha256-cc3f3ab982464e82c0b6a06bc2e2c4bdc5c20df99fbfc225f53981adbcc42bc0`

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
| vector_only | 1 | 1 | 0 | 1 | 1 |
| graph_prior_only | 1 | 1 | 0 | 1 | 1 |
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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-cfb1d94a577129d4d3443f4e0e588167e5df8247c7459669699a79d5c108e8cf |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-87a2303b14491b11370453ad3779c841a2285fada330c8c3944f65e4f731f69f |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-5cb5f8041d64489f7ea4a5ec8dee9b6090a399ca98e4b8dac757c861ad4deafc |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-2dc627db8b5bf5d0853a86a91b1aff982ee225565a0ccda07c33e569b0c07ccc |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-8ac5398b | sha256-78debedf9e90d84795bc37ad1731c2660612e0a0ae1422a0a8f63eec7ef3c5f9 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-8ac5398b | sha256-78debedf9e90d84795bc37ad1731c2660612e0a0ae1422a0a8f63eec7ef3c5f9 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-8ac5398b | sha256-78debedf9e90d84795bc37ad1731c2660612e0a0ae1422a0a8f63eec7ef3c5f9 |
