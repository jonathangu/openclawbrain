# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-176`
- winner mode: `graph_prior_only`
- trace hash: `sha256-480f373d763bedd2f5766cb9a1a8860701112223bd910911c4830c2fc4277912`
- fixture hash: `sha256-2055d04a6856d7cf43d112e858be3651b8402ee14faf73331e6f59144245384e`
- score hash: `sha256-09fe83d760c70c03798a8a4a7471da2c0a55041891ed407d45b59447c8f35725`
- bundle hash: `sha256-76fed809c531ecafd0e273fc79ba2a6c92a825d01d49849be03f7ec9f0604b64`

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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-cc6a8a76ddb7a25937feec38e19ee175087db4867c24970d2759b39f1c9b4bd1 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-4011011688dd6d93918a5836a5bebbb3c7268de9d80f8ccb15203d59c1722d27 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-f274d149f39efb780022e8f045c632710c7c1b765f26afd57316b203fa1b67c9 |
| learned_route | 1 | 1 | 0/2 | 1 | 0 | 1 | 0 | 2 | sha256-64b8f772c2f1e50ccda1465dfebf38eec771d742a57250c652f34ddee5001e41 |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-2f82d349 | sha256-85286930a70aa53ecbdee15e9a9648135b66d268e485757c9e7049f5d80aa6d1 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-2f82d349 | sha256-ad989143a19d0dd096b04434c9b21d2e57f73f930bd42995c7a36bc5d280d9cb |
| learned_route | turn-1 | 40 | yes | 0/2 | yes | no | pack-14182214 | sha256-5fcd1daab419698fa6fa8bd390ec708a1846a010b491a0c552110ad795440949 |
