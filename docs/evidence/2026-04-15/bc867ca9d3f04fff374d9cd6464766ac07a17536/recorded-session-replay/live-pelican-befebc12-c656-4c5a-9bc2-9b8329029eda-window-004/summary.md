# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-befebc12-c656-4c5a-9bc2-9b8329029eda-window-004`
- winner mode: `graph_prior_only`
- trace hash: `sha256-974c0caac77d24b74750a03083f2fe960327dbab94f044b1f352645b0c8977ef`
- fixture hash: `sha256-5332e2f75d9b84ce32dc4225385441cf4e1fdff1345733b1234e8eeb65449d9c`
- score hash: `sha256-67b9db855850d66b1d112b16e8242e94378237da4962473e7ca0720e967a3776`
- bundle hash: `sha256-09389a6c6a1284c8ec05f65fc0e108df895cadd76771256115ab3d31917111a7`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-dbc3f90d1fd6323456b57dfe2268d1b7eda59d985039bca3aa485da45edbfc64 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-1a61f83f162258ddfb983a1556e4f0708d64a634eacaa6ae8de151a642b759ce |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-0c8655ce1df2bbc7e047909c5768fd75e1786ec2e52991ed2702bc1e7756c38a |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-4042d70bfbf2cd6cc46f9df32a29a83044393350a68384389fcfcd488bca484d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e092f506 | sha256-4239ecb1a27dd3c2b8c4028fec12ecebe14b54588b58cdf7b25bc766cf63aa84 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-e092f506 | sha256-c2d03af0244f671cb18636480f42e65604f4c01666ebb75aa566ab939acf5edb |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-e092f506 | sha256-4239ecb1a27dd3c2b8c4028fec12ecebe14b54588b58cdf7b25bc766cf63aa84 |
