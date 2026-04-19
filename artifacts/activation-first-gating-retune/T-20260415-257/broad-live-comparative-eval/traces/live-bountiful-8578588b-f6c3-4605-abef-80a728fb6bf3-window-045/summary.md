# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-045`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8e3583bf5dea5a97f411db7a93b626a380f7abb1e46210bd20e8e3bd2bbad8fe`
- fixture hash: `sha256-3b9683750d86ab2808adcd8363ab4f3221db9cceab2259c7bc66ec4c98677b32`
- score hash: `sha256-ae1b3bba69b1ba97af1d038f919e054f6e642e9e05ed34435447188c414a3b0b`
- bundle hash: `sha256-27f823a5d7846a35480bd5b7f883286d03f5d18fc5a6af6aea3092ae6f809ad2`

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-19530c873b308f4dd0b2f26574b5efb1eeec52061eb993ae949797bdd7b8e58c |
| vector_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-eaece58636d76802bbca9b87d259e82b4623d44fef17040c116bae8422657a52 |
| graph_prior_only | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 1 | sha256-f1e5565552f7dab7e3f4ff6553de6ea08cf92beec8043228470910c83372a0d0 |
| learned_route | 1 | 1 | 0/3 | 1 | 0 | 1 | 0 | 2 | sha256-8637d0c0703e5bc7f4e13150f3d0ccd59de74198851e6a7895872002922d20e5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-4b6d18ba | sha256-ecba964f4f594822628da5a2d406d5336a59d2d1dcb401a02dbaa281a04a52b1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | yes | no | pack-4b6d18ba | sha256-b5fb0bcc8b2993c6ad46087e60aa9acf69004aceb5a86aa808a6a9cd0fd5d0d6 |
| learned_route | turn-1 | 40 | yes | 0/3 | yes | no | pack-4b6d18ba | sha256-ecba964f4f594822628da5a2d406d5336a59d2d1dcb401a02dbaa281a04a52b1 |
