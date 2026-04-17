# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-040`
- winner mode: `graph_prior_only`
- trace hash: `sha256-cb2cad08b75a4c5a17135f849a796fb61b6aa111c1915e1e3fbc43ee29768a21`
- fixture hash: `sha256-84cbdf3fb211d244e4240d521b9374e489cde8de517963996815de9de3b7ff1d`
- score hash: `sha256-ec516ad1fa2847d8c98d85caf4790f17aeb107c2631ea9c928c962b581a5e204`
- bundle hash: `sha256-c8afc97ded94832099f8240e51568fb0d07882a366e72b7a0a29b913b5dc98e9`

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-ed7632976503f51e965dfa45a2710170b9804510f9f71a0462c48d22b4bb5cb9 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-546d7bd41a79889a1a087ba516b6ed1ee2ce6400f165b6c651ea50225fc59a8b |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-b9e96fd2fcecbe03f17883cf4f18283055fc40f91e5bb3ec7b58d66b47288c2b |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-404ad054d368efdf964b2f9015dbe777a4bb3802329f7982fb06ee7d535cc9c4 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3bd689ba | sha256-9bd012677eb2c7ebd91744d0d77c050e357285b9f15ec09c148d0de3cd0c63d7 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-3bd689ba | sha256-8e9feb452e8615e9f21a4bdeb1a50403d90a082e655efc5f0de70f4dfdb18ade |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-11dd01ab | sha256-72335612fa5dd597ba9d8b2f86598d9b0c752a5314e993c9e413486bfb1729cc |
