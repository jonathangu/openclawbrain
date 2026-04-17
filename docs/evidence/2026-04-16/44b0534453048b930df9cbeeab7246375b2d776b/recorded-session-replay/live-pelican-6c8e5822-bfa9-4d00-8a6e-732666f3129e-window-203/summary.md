# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-203`
- winner mode: `graph_prior_only`
- trace hash: `sha256-304f24fec2ef73f307ee3b3bfeff3d4bd90894e7d9fa693794ad2f916befa2ce`
- fixture hash: `sha256-0d0f5f0a3dfc50799aa0a0583bb1e17204f3f01f50323b91030ad8276022d234`
- score hash: `sha256-e08e71eb2e778db5b3e78903897d799be0bde422ad3a64c1a29b471e9808b31b`
- bundle hash: `sha256-4d5e6115fee3c406c4e69727709caf41ee02c821a1ad1c5350b93c31dd94c16b`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-c2db02be6aed4a54b1a82eb4486629fc6a8c812b69fc8bb1feef7e61858ba9b3 |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-a5f30484ae9cce75d60179287a4316b2b950e6d40420667850d7d20cd1abb100 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-d298157ed42326de6098bb21da5393382a4ba9101111354109e2ba2d134adb06 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-8f6ad48d983403441f92166898f952f5d2ce1ea5e4fd50bc36cb98c4c533db0b |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8056ab0b | sha256-35c94b840147dc4c315b4c3b48c72e5ff8a9aab866f9d36a009b2b0f7ed9de92 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-8056ab0b | sha256-41864d1216904505d021ad832af1fd5898b19a438fd2ccf8c7221f7326b799b1 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-7c28ae72 | sha256-89273c231ae4728ff54a3bcdb886b24ee064f668cd5f47422cecee3be2474921 |
