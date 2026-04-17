# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-268c1732-aacb-4a41-b1d1-028262bee45f-window-008`
- winner mode: `graph_prior_only`
- trace hash: `sha256-821174c9e8199055ec82b211fb2d09a993a3651df37b6c8cbf4707f78e0465ca`
- fixture hash: `sha256-bcd47938b5aaf944e8ae47149b98261af7f5e6b78cefab4ab8c21ef4d0f8288c`
- score hash: `sha256-bbacf82a33caf6ef32730e282bb7c22781c5849d00487326adb139eba8fc6eef`
- bundle hash: `sha256-1c41ac14f2aa15ae49d0f45bb9cdafd37212f6131d6641e0f29437a53b9fb5b1`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-46bef670c7b3d0d500dbccdbaeb44127bcccbda5425d78ea64b9256410c95a9e |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-2647121b5f417b367cb7f25c636495ca7092450eba3e93fd908184cecf98a133 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-a5e6590514da80a0a9a7c059a586f01bbb2c6e5da106b5b2be8d80c07d4dc5f0 |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-c03870feaa8fc67e6cfb74c963cdac60df7752fae1384c02dfd2a0bfce5498f5 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c916845e | sha256-d1969c4829ea4d9922bb5ce2f167051c7a8a71279af9d3a86bafef94706c24b2 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-c916845e | sha256-ba549c52688a5441da0395efc24c286c523702c5111fd536a418b2fdb92a93a0 |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-e0d81ccb | sha256-2bfd84ba1f09c7a53c9351a8d5b6474c7f97f7951f7d9c2bcedf0197f6f9e70e |
