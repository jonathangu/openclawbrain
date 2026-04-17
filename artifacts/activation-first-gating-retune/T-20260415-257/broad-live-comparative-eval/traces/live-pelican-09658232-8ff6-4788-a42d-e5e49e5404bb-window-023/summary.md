# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-09658232-8ff6-4788-a42d-e5e49e5404bb-window-023`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f60100eb1742bfc6c299af2f5afe9b6c211473ff986af1ecb211b198ec2ac6e1`
- fixture hash: `sha256-7060077aa89ea2d2ed121c14a4166c1764801c149a1d2df1467761d22c2169ae`
- score hash: `sha256-36f5cebb711ba1fec66af10325de037cd8d4cf92b55ee2116e42d883f6eab60e`
- bundle hash: `sha256-b0b651136b80e1456c305f12f34cfa7f3a8b7a615cf2b426f8f91932302a7d09`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-1af67f3f0f2a5d2c63ece4b570453604d2bc85441d7219830f849b19b9d0d604 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-4630ca6f4e7975b0308dbddcbbb1cd8ea4bee89da009c366ad22172e9952bcf3 |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-6b37028ea2de79d238fbec2e7fca9a650c17c38b26716483cdd3dfcae257de2f |
| learned_route | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 2 | sha256-3dc0907d272b755282bc6210dda42410c5fc6994d73e8cdd8dd35a84275a587a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b71ba40f | sha256-7237a4255a0171dcd2edeb6440e7b112bf2c1fd4605460528f039b4503e64a65 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-b71ba40f | sha256-585c1a597e6c3e42f361ec6a9f88235298607881b64fa47e243ef6fbd1849e4f |
| learned_route | turn-1 | 40 | yes | 0/1 | no | no | pack-39fad4b6 | sha256-78e5921ba4aa4e99af19912996531325b80492d17729bd87e7e891581a036913 |
