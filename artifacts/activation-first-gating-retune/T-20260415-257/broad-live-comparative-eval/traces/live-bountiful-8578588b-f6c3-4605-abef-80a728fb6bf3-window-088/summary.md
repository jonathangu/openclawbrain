# Recorded Session Replay Proof Bundle

- trace id: `live-bountiful-8578588b-f6c3-4605-abef-80a728fb6bf3-window-088`
- winner mode: `graph_prior_only`
- trace hash: `sha256-d408fd5085bee42b21f0981a6a132c6f5610bc4fbc2c34be57ee02be1d61a0ce`
- fixture hash: `sha256-862ded90e7a70c4a33516862a8e1e39d367470070e2af97860bbd4bfdf5f11df`
- score hash: `sha256-1a9287223eabea9c9e072721dd65ae7404c1a77d370a551d6a5d6e46ed72b54b`
- bundle hash: `sha256-038b1d41d70675658fa60426ab54d4976583d1e3a2b7f8154fc9989dce2fb848`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 70 |
| 2 | learned_route | 70 |
| 3 | vector_only | 70 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 3/8
- phrase hit rate: 0.375

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0.5 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0.5 | 0 | 1 |
| learned_route | 1 | 1 | 0.5 | 0 | 1 |

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-608f5e8e26bfe8b9bd3aa5093cf247c400a3d35bb3889f3787f54ae23dbaa484 |
| vector_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-91d22e3c0a70ccddaa97babb8ba4db6b0d7e8735cdbc48db8f507dfbd8cad217 |
| graph_prior_only | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 1 | sha256-3b7e2b9358d6d2da1cee91ace304f78104a86089fd781840e64b149c8c47baea |
| learned_route | 1 | 1 | 1/2 | 0 | 0 | 1 | 0 | 2 | sha256-e3d33526bd0a8aba9942a39f2f3e7d794ed0037888ed15892abf09ebab38c4f3 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 70 | yes | 1/2 | no | no | pack-6840b310 | sha256-c0ac382f2204ff05f8922acfeccbc52b91aa6e937adea23690f18f35468a4d6e |
| graph_prior_only | turn-1 | 70 | yes | 1/2 | no | no | pack-6840b310 | sha256-8473fa9752e6a6711ba5541bf4ee228170b31f94cbf8b8f54a540d9b0f8fe6ba |
| learned_route | turn-1 | 70 | yes | 1/2 | no | no | pack-7edaab49 | sha256-f9f8fa000fdf6e437dc2476a599ddd098b76fb8cfce40487f88f31ad59eb337f |
