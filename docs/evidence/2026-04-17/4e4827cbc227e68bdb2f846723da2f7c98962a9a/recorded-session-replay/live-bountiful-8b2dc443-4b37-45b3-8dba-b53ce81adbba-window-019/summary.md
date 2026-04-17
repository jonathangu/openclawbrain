# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8b2dc443-4b37-45b3-8dba-b53ce81adbba-window-019`
- winner mode: `graph_prior_only`
- trace hash: `sha256-b26af2b1e36bf39a5b818412cda88ed6aba582667f9a54ce799e21e291662727`
- fixture hash: `sha256-dc5c60cd5ff0fd0eb8ea43eb629625260e32638ca4678441b2528e3ed52617bf`
- score hash: `sha256-fcb20c1f57c7a61aab92cfe0f69502b624445d93b8fa57422ae47f760f91fdbd`
- bundle hash: `sha256-92e1e9f1ea835855fa05dcffea6d4bac84e853bd2cf2b65a432f4ea9b1942913`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-8233e1dc85a16271682dd831a32fd53162f821cae19b4a63ef88dbd637e3c9f6 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-830bb469b3a907d9fa4fcbefedb60483e17955294dda5d82da4b4b7fb3208747 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-eb5a09e0b920be5a5a31abcbe41146b0190d100a8c4881236472f8df9298182e |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-df55f95c2f21fddfa321ff5ac0c009ce6e9ed4284900f4b3661f5ac799a5271c |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-9cbe6d2e | sha256-c42125a7142f24df4eeddf44756ebd0ee4ffba80dd729cf6e7f73d85b13accc5 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-9cbe6d2e | sha256-f87706f1c09fc6fa9fbf4624e60a557d27680e3ea38ca4d471ecc8845680bdc6 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-7a37dbdd | sha256-904a2b9753f6bba683776a5d907e88febb4ead03c70c2977feca6c7251e383a5 |
