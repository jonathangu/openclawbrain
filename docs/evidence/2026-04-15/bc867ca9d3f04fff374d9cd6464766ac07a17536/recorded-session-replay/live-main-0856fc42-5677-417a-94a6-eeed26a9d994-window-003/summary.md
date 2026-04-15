# Recorded Session Replay Proof Bundle

- trace id: `live-main-0856fc42-5677-417a-94a6-eeed26a9d994-window-003`
- winner mode: `graph_prior_only`
- trace hash: `sha256-8112927457240059417bedc3d26ba052a003896d620c2316ad6b12373ef80eef`
- fixture hash: `sha256-14ad40161fa5c35ed07d9d394829c949bb081beaa26c47469b137af3b630df8b`
- score hash: `sha256-0300697c858884e7c63e1c60d611b4d9281bae134ec2def892d70809add1cb99`
- bundle hash: `sha256-b1f01f123b5ca0ad624dfa3ba70b192cc5a80b150367d5726d919e5126ad343a`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 100 |
| 2 | learned_route | 100 |
| 3 | vector_only | 100 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 9/12
- phrase hit rate: 0.75

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 1 | 0 | 1 |
| graph_prior_only | 1 | 1 | 1 | 0 | 1 |
| learned_route | 1 | 1 | 1 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7e54eea5dd476d45e5e7ab52a9b0ed2c646fc990677d2858d9966f3baecd8936 |
| vector_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-9a8fcc87ce035aa5a6c1767d04514a46bc9e38741a8c51aab036b77d61bc3f9f |
| graph_prior_only | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 1 | sha256-309239146b86d8e0a086079f38ce108b3f7e2da5e61757ee56e1df48de844cec |
| learned_route | 1 | 1 | 3/3 | 0 | 0 | 1 | 0 | 2 | sha256-3f6c1625c4fc1fc08ef55bb7359529e02ed96387e1eef5c11fcfe27388d63761 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 100 | yes | 3/3 | no | no | pack-d36c8a0e | sha256-c72f3719c3de737e3dd5465caa4ed0ea40c6805db11f8af101f96d57214aaf3e |
| graph_prior_only | turn-1 | 100 | yes | 3/3 | no | no | pack-d36c8a0e | sha256-d77d15fa034c59f20971a6eb5b8e1866c1a60d770a4207ad1e395b36e512ae4c |
| learned_route | turn-1 | 100 | yes | 3/3 | no | no | pack-d36c8a0e | sha256-c72f3719c3de737e3dd5465caa4ed0ea40c6805db11f8af101f96d57214aaf3e |
