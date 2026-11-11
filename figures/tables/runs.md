Runs in `experiments/latent_sft/results/<campaign>/<run>`. Tinker checkpoints carry a 7-day TTL, so the paths are a record, not something that still loads.

| campaign | run | stages (size: passed) | final checkpoint |
|---|---|---|---|
| curriculum | `latent_a5_oss20b_curriculum` | 10:y, 15:y, 20:y, 30:y, 40:n | `tinker://d78fdc53-bcbc-575d-8220-9bb0a4b2d510:train:0/weights/final` |
| curriculum | `latent_a5_qwen_curriculum` | 10:y, 15:y, 20:y, 30:y, 40:n | `tinker://68e454d2-6d80-50d9-9a93-fdb87afc9600:train:0/weights/final` |
| curriculum | `latent_arith8mod_oss20b_curriculum` | 1:y, 2:n | `tinker://f462a9ee-d25d-5477-812e-d9c4e23724bf:train:0/weights/final` |
| curriculum | `latent_arith8mod_qwen_curriculum` | 1:y, 2:y, 3:y, 5:n | `tinker://4eb1311a-53fd-54ca-bdb6-80aaa9ece2cf:train:0/weights/final` |
| curriculum | `latent_chain_oss20b_curriculum` | 4:n | `tinker://0d405bfe-747f-5c63-91cc-6a858ec1e8a2:train:0/weights/final` |
| curriculum | `latent_chain_qwen_curriculum` | 4:y, 8:y, 16:n | `tinker://b8adefeb-0db9-5901-98c3-e57e09adb6ff:train:0/weights/final` |
| curriculum | `latent_fc_oss20b_curriculum` | 1:y, 2:y, 4:n | `tinker://0d1144d9-6794-5b75-b146-1a8e2b7c4b4b:train:0/weights/final` |
| curriculum | `latent_fc_qwen_curriculum` | 1:y, 2:y, 4:n | `tinker://f34e8c5c-430c-5630-9073-b8f847999e56:train:0/weights/final` |
| curriculum | `latent_gl32_oss20b_curriculum` | 10:n | `tinker://377c3063-c6ee-5be0-8da4-6ef9a312d1e1:train:0/weights/final` |
| curriculum | `latent_gl32_qwen_curriculum` | 10:y, 20:y, 30:y, 40:y, 60:y, 100:n | `tinker://3734f3bf-56d1-5fa6-8498-ff16f3545ce8:train:0/weights/final` |
| curriculum | `latent_horn_oss20b_curriculum` | 8:n | `tinker://000c5470-5fab-53e5-83fd-1a0216c59b79:train:0/weights/final` |
| curriculum | `latent_horn_qwen_curriculum` | 8:n | `tinker://9e894ece-220f-51fd-8814-bdc393731b30:train:0/weights/final` |
| curriculum | `latent_mod2_oss20b_curriculum` | 10:y, 20:y, 40:y, 60:y, 80:y, 100:y | `tinker://62e9e114-83a2-572f-bed2-904d75cfb748:train:0/weights/early_stop_1` |
| curriculum | `latent_mod2_qwen_curriculum` | 10:y, 20:y, 40:y, 60:n | `tinker://dd0faab5-f99c-52f8-bb4d-7b20ad0a6fe0:train:0/weights/final` |
| direct | `latent_a5_oss20b_direct` | 100:n | `tinker://1dcb6fb7-7516-5d30-b001-7f501f808fe6:train:0/weights/final` |
| direct | `latent_a5_qwen_direct` | 100:n | `tinker://e86769e7-dc9f-5a44-b92f-bd6c39f9c075:train:0/weights/final` |
| direct | `latent_arith8mod_oss20b_direct` | 100:n | `tinker://3d4a203f-3e8c-56a2-9794-43312e03c0d0:train:0/weights/final` |
| direct | `latent_arith8mod_qwen_direct` | 100:n | `tinker://8c87504e-681b-5782-83ab-6fe80dd326e2:train:0/weights/final` |
| direct | `latent_chain_oss20b_direct` | 50:n | `tinker://d0cda5f2-6d43-59a3-b3a8-8e06360e331a:train:0/weights/final` |
| direct | `latent_chain_qwen_direct` | 50:n | `tinker://4b9f0d71-5fe2-547b-8ac3-4c0d40cd947e:train:0/weights/final` |
| direct | `latent_fc_oss20b_direct` | 25:n | `tinker://7769e7d4-9abf-5de7-bf7b-c8d3225a1913:train:0/weights/final` |
| direct | `latent_fc_qwen_direct` | 25:n | `tinker://54f40ed3-a1a8-50ba-94a0-3762660ec856:train:0/weights/final` |
| direct | `latent_gl32_oss20b_direct` | 100:n | `tinker://6ac751a0-69aa-5aa6-8c59-bcd217abce62:train:0/weights/final` |
| direct | `latent_gl32_qwen_direct` | 100:n | `tinker://d5a77d2c-c9ce-5993-9dbe-79e1d7bb43bb:train:0/weights/final` |
| direct | `latent_horn_oss20b_direct` | 50:n | `tinker://9ce4bfae-0b8f-52f7-8ad2-4dc72e36699d:train:0/weights/final` |
| direct | `latent_horn_qwen_direct` | 50:n | `tinker://27960285-fc85-5e9f-98f0-35c0986b0955:train:0/weights/final` |
| direct | `latent_mod2_oss20b_direct` | 100:n | `tinker://38da225f-a2d4-51a8-9ebe-658e68642106:train:0/weights/final` |
| direct | `latent_mod2_qwen_direct` | 100:n | `tinker://952200f0-67ab-5536-936c-1305cbd668e6:train:0/weights/final` |
| comparison_method | `latent_a5_oss20b_ansonly_seed1` | 10:y, 15:y, 20:n | `tinker://4e5005e0-a26b-502e-8edd-54eb8acc8cdc:train:0/weights/final` |
| comparison_method | `latent_a5_oss20b_ours_direct_seed1` | 30:n | `tinker://333e0c46-5d2b-5d81-8d92-68c906ce4dea:train:0/weights/final` |
| comparison_method | `latent_a5_oss20b_ours_seed1` | 10:y, 15:y, 20:y, 30:n | `tinker://4a86aac4-a960-5cf2-a181-1b8ca88a7990:train:0/weights/final` |
| comparison_method | `latent_a5_qwen_ansonly_seed1` | 10:y, 15:y, 20:n | `tinker://0fdeb923-de0e-5d1d-8170-5a432e2666af:train:0/weights/final` |
| comparison_method | `latent_a5_qwen_ours_direct_seed1` | 30:n | `tinker://1933a402-a3c0-5921-9a58-f9a34aeed6cd:train:0/weights/final` |
| comparison_method | `latent_a5_qwen_ours_seed1` | 10:y, 15:y, 20:y, 30:n | `tinker://a69e526f-504e-526e-997c-9d46e3f9cef4:train:0/weights/final` |
| comparison_method | `latent_gl32_qwen_ansonly_seed1` | 10:n | `tinker://cd29a63f-1eec-565a-b2ca-6333555641bb:train:0/weights/final` |
| comparison_method | `latent_gl32_qwen_ansonly_seed2` | 10:n | `tinker://f0af72e3-ca72-53e7-b28e-49995d664a54:train:0/weights/final` |
| comparison_method | `latent_gl32_qwen_ansonly_seed3` | 10:n | `tinker://9ce57ba6-1bba-5573-a6ba-6a4dde17c6b8:train:0/weights/final` |
| comparison_method | `latent_gl32_qwen_dotbydot_seed1` | 10:n | `tinker://9c8acd7f-2b31-5de6-a530-07d8a9840cb4:train:0/weights/final` |
| comparison_method | `latent_gl32_qwen_dotbydot_seed2` | 10:n | `tinker://1e4d432c-3402-5457-85a2-e936fb330528:train:0/weights/final` |
| comparison_method | `latent_gl32_qwen_dotbydot_seed3` | 10:n | `tinker://bf1a6268-1191-51c3-ac36-a003802a16b0:train:0/weights/final` |
| comparison_method | `latent_gl32_qwen_dotbydot_seed4` | 10:n | `tinker://f973053f-ca05-533f-bec5-fdfc1548b0e2:train:0/weights/final` |
| comparison_method | `latent_gl32_qwen_nocot_seed1` | 10:n | `tinker://10c65dbe-8e8c-5192-aa6e-0468914c5a65:train:0/weights/final` |
| comparison_method | `latent_gl32_qwen_nocot_seed2` | 10:n | `tinker://607979d8-0364-5127-9eae-24f2d85ae36f:train:0/weights/final` |
| comparison_method | `latent_gl32_qwen_ours_direct_seed1` | 40:n | `tinker://9d198b0d-13c9-5848-8d2e-d430afa4b0ce:train:0/weights/final` |
| comparison_method | `latent_gl32_qwen_ours_direct_seed2` | 40:n | `tinker://fda1d22d-3976-51e5-b12a-06b2c681d73a:train:0/weights/final` |
| comparison_method | `latent_gl32_qwen_ours_direct_seed3` | 40:n | `tinker://695d15fa-fb50-58b0-b0aa-62b9a050510a:train:0/weights/final` |
| comparison_method | `latent_gl32_qwen_ours_seed1` | 10:n | `tinker://02c6b5d2-bc4d-5327-a089-da9659ea9146:train:0/weights/final` |
| comparison_method | `latent_gl32_qwen_ours_seed2` | 10:y, 20:y, 30:y, 40:y | `tinker://8ce98010-9f5b-5f56-8238-733f9cfa822b:train:0/weights/early_stop_5` |
| comparison_method | `latent_gl32_qwen_ours_seed3` | 10:y, 20:y, 30:y, 40:y | `tinker://dbac0c61-640a-5fb1-8bfb-210406e00822:train:0/weights/early_stop_6` |
| comparison_method | `latent_mod2_oss20b_ansonly_seed1` | 10:y, 20:n | `tinker://4a764314-97c6-5e44-8b9a-9b18bdb2a75a:train:0/weights/final` |
| comparison_method | `latent_mod2_oss20b_ansonly_seed2` | 10:y, 20:n | `tinker://b3210348-cc4c-5c39-b3a4-dcfc47055c76:train:0/weights/final` |
| comparison_method | `latent_mod2_oss20b_ansonly_seed3` | 10:y, 20:y, 40:n | `tinker://c941b95e-8a52-529a-b607-c804ed50161f:train:0/weights/final` |
| comparison_method | `latent_mod2_oss20b_dotbydot_seed1` | 10:y, 20:n | `tinker://512fd014-414e-56bd-912b-ce6708ea994a:train:0/weights/final` |
| comparison_method | `latent_mod2_oss20b_dotbydot_seed2` | 10:y, 20:n | `tinker://7a8938e2-e3ca-5dc5-b694-330e308a9c69:train:0/weights/final` |
| comparison_method | `latent_mod2_oss20b_dotbydot_seed3` | 10:y, 20:n | `tinker://d0e5a3ce-3f34-59df-a2f7-0794ce166c56:train:0/weights/final` |
| comparison_method | `latent_mod2_oss20b_nocot_seed1` | 10:y, 20:n | `tinker://175ffc43-e402-5b28-bb40-07965db948b6:train:0/weights/final` |
| comparison_method | `latent_mod2_oss20b_nocot_seed2` | 10:y, 20:n | `tinker://e347b206-be62-5585-992c-3ed3f2d964c4:train:0/weights/final` |
| comparison_method | `latent_mod2_oss20b_ours_direct_seed1` | 40:n | `tinker://fe0549a9-fb14-537c-82d8-ac641648a587:train:0/weights/final` |
| comparison_method | `latent_mod2_oss20b_ours_direct_seed2` | 40:n | `tinker://ef53c9c3-90cf-59e2-845d-d1b8261df813:train:0/weights/final` |
| comparison_method | `latent_mod2_oss20b_ours_direct_seed3` | 40:y | `tinker://36e0c25f-59b8-5507-8228-50f1b1124ae0:train:0/weights/early_stop_12` |
| comparison_method | `latent_mod2_oss20b_ours_seed1` | 10:y, 20:y, 40:y | `tinker://5423d22d-368f-5cef-a431-d1a05289c49d:train:0/weights/early_stop_3` |
| comparison_method | `latent_mod2_oss20b_ours_seed2` | 10:y, 20:y, 40:y | `tinker://e5ef9163-c761-5f71-8081-1770cbbc47dd:train:0/weights/early_stop_4` |
| comparison_method | `latent_mod2_oss20b_ours_seed3` | 10:y, 20:y, 40:y | `tinker://eb2d2a7a-7c35-59f6-b51f-5cc87ecfb0eb:train:0/weights/early_stop_6` |
| comparison_method | `latent_mod2_qwen_ansonly_seed1` | 10:y, 20:n | `tinker://8e27cf10-8e70-5091-95d1-d7e82ded1819:train:0/weights/final` |
| comparison_method | `latent_mod2_qwen_ansonly_seed2` | 10:y, 20:n | `tinker://ec66ee62-6153-51df-bbc2-b30af2534eeb:train:0/weights/final` |
| comparison_method | `latent_mod2_qwen_ansonly_seed3` | 10:y, 20:y, 40:n | `tinker://b04a1356-d134-548c-89f5-48346a4f994e:train:0/weights/final` |
| comparison_method | `latent_mod2_qwen_dotbydot_seed1` | 10:y, 20:n | `tinker://00ab826d-e87b-5b77-85dc-d56608fd6011:train:0/weights/final` |
| comparison_method | `latent_mod2_qwen_dotbydot_seed2` | 10:y, 20:n | `tinker://919c8476-a305-5a87-b329-433193a27973:train:0/weights/final` |
| comparison_method | `latent_mod2_qwen_dotbydot_seed3` | 10:y, 20:n | `tinker://ad4a1e7f-dc3f-595f-9498-3291e8b8fea1:train:0/weights/final` |
| comparison_method | `latent_mod2_qwen_nocot_seed1` | 10:y, 20:n | `tinker://065621e3-e287-56f0-a594-99c8ed731916:train:0/weights/final` |
| comparison_method | `latent_mod2_qwen_nocot_seed2` | 10:y, 20:n | `tinker://5d326836-61e7-563c-b702-2d6a2b20caec:train:0/weights/final` |
| comparison_method | `latent_mod2_qwen_ours_direct_seed1` | 40:y | `tinker://3b5f9c31-f559-501c-9359-c1bd42abadd2:train:0/weights/early_stop_6` |
| comparison_method | `latent_mod2_qwen_ours_direct_seed2` | 40:y | `tinker://ea04f37d-8778-5d20-8d21-c4d9421cd151:train:0/weights/early_stop_17` |
| comparison_method | `latent_mod2_qwen_ours_direct_seed3` | 40:y | `tinker://f28cdc3e-5fd8-5344-b63d-39230bf08be5:train:0/weights/early_stop_4` |
| comparison_method | `latent_mod2_qwen_ours_seed1` | 10:y, 20:y, 40:y | `tinker://97bd2b3d-02c4-5a8e-ad29-2cd8e1587e19:train:0/weights/early_stop_2` |
| comparison_method | `latent_mod2_qwen_ours_seed2` | 10:y, 20:y, 40:y | `tinker://c79033d8-d7e9-539d-85d3-ed5d982bc58c:train:0/weights/early_stop_2` |
| comparison_method | `latent_mod2_qwen_ours_seed3` | 10:y, 20:y, 40:n | `tinker://5c2353cb-ba9d-5e00-aeb5-938629e54f0d:train:0/weights/final` |
| nocot | `latent_a5_oss20b_nocot` | 10:y, 15:y, 20:n | `tinker://7042d3f1-6f6a-5365-a05c-b24d8b129111:train:0/weights/final` |
| nocot | `latent_a5_qwen_nocot` | 10:y, 15:y, 20:n | `tinker://70ca0fb3-a2c3-5b75-b4b4-cd30557ea8b5:train:0/weights/final` |
| nocot | `latent_arith8mod_oss20b_nocot` | 1:y, 2:n | `tinker://a2358c62-90ee-5c8b-aac6-09dc16b3c589:train:0/weights/final` |
| nocot | `latent_arith8mod_qwen_nocot` | 1:y, 2:n | `tinker://73855e26-d333-5c1d-aa0e-f8b6e2a61440:train:0/weights/final` |
| nocot | `latent_chain_oss20b_nocot` | 4:y, 8:y, 16:n | `tinker://b48d756c-1d0e-524a-aee6-dd7ab1d4c3b9:train:0/weights/final` |
| nocot | `latent_chain_qwen_nocot` | 4:y, 8:y, 16:n | `tinker://93138630-e74b-54dc-9726-4f3726ec6fcd:train:0/weights/final` |
| nocot | `latent_fc_oss20b_nocot` | 1:y, 2:n | `tinker://58fe0181-8a3b-5fcd-ae6a-fe6404921ec6:train:0/weights/final` |
| nocot | `latent_fc_qwen_nocot` | 1:y, 2:n | `tinker://8f9d3409-ed2e-506c-8a83-283400c70528:train:0/weights/final` |
| nocot | `latent_gl32_oss20b_nocot` | 10:n | `tinker://7affb953-be7c-5912-a199-a4279cf87c8f:train:0/weights/final` |
| nocot | `latent_gl32_qwen_nocot` | 10:n | `tinker://7aa4da2a-04f1-59c9-9122-b3df96203fb1:train:0/weights/final` |
| nocot | `latent_horn_oss20b_nocot` | 8:n | `tinker://9c1fe4df-9288-578b-9cd9-d0d711b23081:train:0/weights/final` |
| nocot | `latent_horn_qwen_nocot` | 8:n | `tinker://519fc79b-caf6-51f6-92fa-bd21cbb37783:train:0/weights/final` |
| nocot | `latent_mod2_oss20b_nocot` | 10:y, 20:n | `tinker://97685f87-a6b3-5916-ad7f-15a7a9f590bc:train:0/weights/final` |
| nocot | `latent_mod2_qwen_nocot` | 10:y, 20:n | `tinker://6ff7d067-d33a-555c-afac-15d837ad798c:train:0/weights/final` |
| open | `latent_a5_oss20b_open` | 100:y | `tinker://a94109d7-e6b1-5574-820a-9f7f150f61ec:train:0/weights/early_stop_1` |
| open | `latent_a5_qwen_open` | 100:y | `tinker://b87b8a42-3655-5bf4-9f5b-52e6137f3576:train:0/weights/early_stop_1` |
| open | `latent_arith8mod_oss20b_open` | 100:y | `tinker://132052a9-e1f5-58a6-80bf-1a313e51e6a3:train:0/weights/early_stop_1` |
| open | `latent_arith8mod_qwen_open` | 100:y | `tinker://47d19726-7009-5ad8-9059-f34ac34e22c1:train:0/weights/early_stop_1` |
| open | `latent_chain_oss20b_open` | 50:y | `tinker://388bae4c-eefb-5104-a5c9-a88155355027:train:0/weights/early_stop_1` |
| open | `latent_chain_qwen_open` | 50:y | `tinker://fb4cc354-5efa-56a1-8c6a-7497fcce49d4:train:0/weights/early_stop_1` |
| open | `latent_fc_oss20b_open` | 25:y | `tinker://cb1ac6ad-7a1d-5839-b3b6-299b04c800ac:train:0/weights/early_stop_1` |
| open | `latent_fc_qwen_open` | 25:y | `tinker://4acadfba-cb23-59e2-bab4-c86d1cdbe1e4:train:0/weights/early_stop_1` |
| open | `latent_gl32_oss20b_open` | 100:y | `tinker://6d3ff489-39bc-5dd2-9129-b86299acda7c:train:0/weights/early_stop_1` |
| open | `latent_gl32_qwen_open` | 100:y | `tinker://63447da9-f40b-5bc3-97ad-ade822e59eaf:train:0/weights/early_stop_1` |
| open | `latent_horn_oss20b_open` | 50:y | `tinker://8b09a2f6-72f6-5f0e-8457-7ade61a3ccd9:train:0/weights/early_stop_1` |
| open | `latent_horn_qwen_open` | 50:y | `tinker://0989290f-ed3b-5665-bdb9-7ad004166129:train:0/weights/early_stop_1` |
| open | `latent_mod2_oss20b_open` | 100:y | `tinker://5ea87cb9-734a-591e-9474-0cc0cd963660:train:0/weights/early_stop_1` |
| open | `latent_mod2_qwen_open` | 100:y | `tinker://0ae36ca5-57eb-5262-8240-a06397ac7ce2:train:0/weights/early_stop_1` |
