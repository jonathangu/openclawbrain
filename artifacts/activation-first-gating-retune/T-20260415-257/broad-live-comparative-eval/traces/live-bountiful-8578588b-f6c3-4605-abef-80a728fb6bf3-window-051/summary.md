# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-051`
- winner mode: `graph_prior_only`
- trace hash: `sha256-87081ed5c11e35636d4070ee89bd588bb6f63117995b8bb493b8643be003ec58`
- fixture hash: `sha256-43945ff6ddb3d96c1170069d079615a310c124eadab260a82b171a25d542a6d7`
- score hash: `sha256-05095924e0d140ed86741834cfd6ea97f15e09b502b7827fc9b5bd6912f63708`
- bundle hash: `sha256-1a1b695ddfa215d9ee0fd45f3d001caa4d2ed4d873d373ff68d6d1e46efd8d10`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-427320990f4549fdf09c9b73d9eebc938af8aaf238ccccbd97c3c1df3afef6b8 |
| vector_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-8d2b5b958e558343156572ead2b76ce020badcdad6a00319f6a58909b22a47f4 |
| graph_prior_only | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 1 | sha256-9d41f29f811ba57d32c29fb28d4b60053ce65a20a86878e4d56a0f77f151b310 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-06e09d7aaf03a4c4a64a6841a4a5832e472263dfea6ede62cd6eecf6f47c4a63 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-87708f44 | sha256-1fa8f21eab34191aadd81e65250695e28b2d1074404117b5364a31e2847176a1 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | yes | no | pack-87708f44 | sha256-77e7aa2140e46d6cfc071f2426a29a13286c94522bf3768dcc73ab40c31ebf00 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-87708f44 | sha256-1fa8f21eab34191aadd81e65250695e28b2d1074404117b5364a31e2847176a1 |
