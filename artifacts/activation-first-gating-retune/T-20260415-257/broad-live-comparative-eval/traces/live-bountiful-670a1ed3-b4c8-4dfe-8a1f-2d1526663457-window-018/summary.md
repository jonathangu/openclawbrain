# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-670a1ed3-b4c8-4dfe-8a1f-2d1526663457-window-018`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8731aca670fb1adc2a11de661b208e90de02229e43a59b819be0c26634995543`
- fixture hash: `sha256-b091c6d75f126cd4fa41e0e62e2c1bde2a5cadf897b977dd808714e16a9eb7f9`
- score hash: `sha256-a90d2f2cdfa14c1dce06c3f2d74b747fd506dbdee15ad1a7b38b6cd23545733d`
- bundle hash: `sha256-4d8b705d4be5783b5177ce1c555e5576f975eaa3aec1e6af90fdf027308227f4`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-4dd25e120884595a4500dd8027a1e5e49f93c256e2e2739aa127521c9309576c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-73a49018c42cbc8959ff403849e7cde0b9b26b5aa65a5cee094adb36f1afe8b2 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-77d66ce784f5a1d14936a09f39a7ee68b299e2c700ea2e638cf7a9e215df0441 |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-2be4bbaf8ef62a8e62beb3781cf9b421a85bb47c59216e4353ea64a0b05a9898 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c542c562 | sha256-ee627aabd23b570f1ade9da4756b0c9700d7946baec9f73b6600f6c91616be7c |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-c542c562 | sha256-26054a635afb0d4a364dabaeb428546d6d98015683637cdd7eb0f6e61b957a42 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-0d68d975 | sha256-3c2a01f481a5924195234337d8be0a6c1b8ffeee8c243e7397cfc592fef80c4e |
