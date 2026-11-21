# Source update utility：独立方法审查

日期：2026-11-04。范围为源域离线辅助任务；只读代码和已存产物，CPU 合成数据与已锁定源检查点。**结论：现有候选未通过原 gate；有明确的目标／协议口径风险，也复现了一个特定执行配置下的数值观测问题，但没有证据表明该数值问题导致本轮失败。** 不改变 gate，不声称 J-TEXT 在线、EAST 完整流收益或物理规律。

## 主要发现

1. **高：全局上移是目标允许的解，不是已证实的代码漏洞。** [utility.py:81](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_anticipation_20261104/utility.py:81) 平均各例归一化 BCE 差，没有逐组安全约束，也不把真实反馈相对保持／倒序的优势纳入训练目标。批次取样与 VAL 的按炮、按组评分也不同。合成例中，两正两负、等敏感度、同一 +0.05 logit 移动，总归一化风险变化 −0.37814，但安全例 +0.12186、正例 −0.87814：即使标签平衡，均匀上移仍可降低总目标。**这只证明目标允许该现象，并未证明实际头能或确实实现了常数偏移。**

   独立重算已存逐点评分：主视图 future125 的安全组 +0.001084143、30–125 ms 组 −0.014365467；按炮均值再两组等权后，real −0.006640662、reverse −0.006643387、hold −0.006679401。安全点上移 603/638，旧余弦版本 119/638；危险点上移 132/155。它与广泛上移一致，但不能证明每点移动完全相同。原 gate 的安全改善、优于 reverse、优于 hold、更早破裂保护和 native 视图保护均失败，故总体失败；不得因为某组或平均损失改善而放宽。见 [VAL summary](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_update_utility_val_20261104_v1/summary.json) 与 [gate](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_anticipation_20261104/evaluate.py:54)。

2. **高：及时性教学口径存在明确冲突，是既定设计问题而非暗中未来泄漏。** 用户此前要求 late≤30 ms 不进入主要及时性教学；[设计计划:24](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/SOURCE_ANTICIPATION_PLAN_20261104.md:24) 将其排除在主要及时性分析外，但 [同一计划:44](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/SOURCE_ANTICIPATION_PLAN_20261104.md:44) 又明确要求四组共同教学。实际 [schedule](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_anticipation_20261104/design.py:120) 每批四组各一例；重算 512 抽样，0–30 ms 有 128 次，全部 timely_eligible=false、future125=1，并进入同一个风险效用平均；其占全部 256 个 future125 正例的一半。128 步历史的坐标和该头标签与 schedule/target 全部吻合。评价主 balanced 则仅用 safe 和 30–125 ms。故结果是包含晚期监督的 future125 辅助训练，**不能称为严格排除晚期样本的及时性教学**。建议主进程记录此协议差异，未来若要严格排除，先单独预注册设计和对照，不改写本轮历史或验收。

3. **中：分母没有直接抹掉置信度，但会改变样本权重且弱敏感度可能不稳定。** 令 c=logit(threshold)，m=σ(z_before−c)，g 为诊断损失对临时参数梯度，d=(1−2y)∇z_before。按 [meta.py:142](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_anticipation_20261104/meta.py:142) 和 [utility.py:17](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_anticipation_20261104/utility.py:17)，当步长小且梯度范数远大于 1e−6 时，一阶归一化风险变化约为 −|m−y| cos(g,d)。故相较旧单位余弦，BCE 的置信度因子仍在；它是**相对于阈值偏移后的 margin 置信度**，不等同原概率或已校准概率：固定阈值 0.72589469、原概率 0.9 时 m≈0.77265。分母近似消去初始 logit 敏感度范数，不等价于优化未归一化的风险损失，有限步曲率及不同分组分布也会改变实际权重。

   代码仅排除严格零敏感度，不排除任意小的非零值；平滑尺度至少 0.05×1e−6=5e−8。一个非源模型的弯曲合成例在初始敏感度 1e−9、步长 0.05 时得到归一化变化 12503.90，说明一阶尺度可能放大曲率／数值误差，**不证明真实记录出现了该情形**。建议以后按组报告初始敏感度、proxy 梯度范数、未归一化／归一化损失变化及分母分位数；预先设计同预算的归一化对照，不依据本次 VAL 调参。

4. **中：条件性的零更新数值观测缺陷，未证实影响本轮。** [utility.py:72](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_anticipation_20261104/utility.py:72) 的 before 使用 no_grad，after 使用带梯度的前向。在锁定源检查点、合成输入、辅助头权重和偏置均为零的 CPU 例子里，临时状态严格为零：关闭 MHA fastpath 时 logit 差和 utility 均为 0；开启时最大 logit 差 9.536743e−7、utility −0.000196196。先前审查仅在两个输入上得到两模式相等，不覆盖所有输入。float64 BCE 无法恢复 float32 logit 已损精度。

   本轮[归档训练代码 ZIP](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_update_utility_heads_20261104_v1/code.zip) 的 train.py:182 调 configure，归档 run.py:38–45 明确关闭 MHA fastpath；因此上述 CPU 反例**不证明本次 CUDA 结果受影响**。固定基模／基线时，before 的数值偏移对辅助头是常数，不会仅凭该常数改变其梯度，但会误报效用数值，可能影响外部依赖该数值的判断。后续可在实际后端验证零步、同执行路径 before/after；让基线结果 detach 不要求它采用不同计算路径。

5. **中：当前梯度与因果边界没有发现已证实断链。** [meta.py:149](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_anticipation_20261104/meta.py:149) 用 create_graph=True 求 proxy 对临时参数的梯度，[utility.py:69](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_anticipation_20261104/utility.py:69) 保留实际归一化步的外层路径。固定基模、零起始状态且只训练独立头时，before 与敏感度不依赖待训练头，detach 数学上合理；若以后联合训练基模、初态或选择器则必须重新推导。旧 meta 的 cosine 仍被计算作诊断，但未加进新 utility 的总损失。

   重算 V2 包：TRAIN 10,044 条／1,303 炮、VAL 946 条／145 炮，均为唯一 shot/query，maturity=origin+25、origin+26≤query≤origin+50 的违例均为 0。[归档 train.py:131–148](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/probe_outputs/source_update_utility_heads_20261104_v1/code.zip) 先读源炮再截取 origin 前缀、已成熟 25 点与 query 前缀；这符合离线切片因果边界，**不等于真实流式 I/O**。未来125结局标签是离线外层监督，不得进入部署内层更新；[VAL 代码:238–260](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/source_anticipation_20261104/evaluate.py:238) 在预测封存后才用于评分。该结论是对保存代码／记录的核查，不是对运行过程的独立逐 I/O 监控，也不保证完整炮内多次累计更新。

6. **低至中：数值与记录解释边界。** [forecast_loss:142](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/simple23.py:142) 跨 batch 全部观测点归一化，训练 B=4 而局部评价 B=1；[normalized_step:12](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/experiments/risk_aligned_online/fast_update.py:12) 的 epsilon 使微弱梯度时两者不严格尺度不变。合成梯度范数 1e−6 与 2.5e−7 时步长分别 0.0353553 和 0.0121268；另一个四例锁定源模型合成输入中，B4 与逐例状态最大差仅 6.52e−9。真实弱梯度发生率未知。训练日志的 risk_delta_mean 是全部行均值，而 utility 只平均有支持行；本轮 384 个头步记录均为 4/4 支持、无零 proxy／零 desired，故此口径差异未在已记录批次触发。未归一化均值本来也不能直接当作归一化目标。

## 对照缺口与必要建议

- 保留原 static、native、real/reverse/hold、时间打乱及两视图 gate；对真实与假反馈作同炮配对，报告分组 logit 差、尺度分布和安全／危险方向。均匀 logit 移动只能作为**预先指定的解释性反事实**，不能代替模型、改门槛或通过 gate。历史时间打乱只改变 56/512 次曝光，且各视图／时距的正例暴露不完全相同，见[先前拟合审查:15](/media/hetaisheng/044A81D94A81C83E/Fuison-intra-trajectory-20261028/SOURCE_ANTICIPATION_FIT_AUDIT_20261104.md:15)。
- 检查 query 之后、未用于该步更新的留出诊断预测，与原生头和保持基线比较；拟合该步的 proxy 变好不能证明对未来诊断有预测力。之后若研究逐组约束、真实减假反馈或归一化消融，应作为新的固定预算实验，保留原模型和原验收，报告各组而非只报均值。
- 用户另提供 EAST20 局部结果：source_decoder_absolute 的 safe/real-v-hold/reverse 亦失败，800 条 native 参考与旧运行 exact。本审查**未独立读取或复算**该新增 EAST 产物；它不是完整流或独立成功证据。

## 本次验证、定位与限制

- 独立读入两轮已存 scored.json，按 (view, head, mode, shot, query) 唯一配对：22,704/22,704 before logits 完全相同，5,676/5,676 native-reference after logits 完全相同；逐炮重算上文 real/reverse/hold 和安全点计数。旧、新 scored SHA-256 分别为 3c1cf3a3e06dff18b99c36f5fa577e16290c8a7eaeefc70b1260a9ba8e6edab3、0d68fb757f1402e8d44e1b898d2806f3d84315219689a607ce2be5ee2c260f2e，均匹配各自 completed.json。新 summary 和 comparison 也分别匹配完成哈希 01d49698faf77b52862b53e68ed8845992081cc28902fe28464be7073d2d74f3、510e0b8134c2a339bb17bb58d615f9e22c35d10e5b434ee3af181ac0a7d1c052。该复算检验保存值而非重新运行神经网络。
- 拟合 ZIP SHA-256 822de558055087498b9f903484dba214d2955ab1bd058c3974d85e19aaf716f3，匹配完成记录；其中 utility.py 和 meta.py 分别为 337aeb4dba32d2958b2b7ed3ca7a6b0f63be5507bbfa3fa208470f5c7355d235、c7804fe5765de3c6fbfedff63be406df8e7a77e55dabe94ce60112d81bb7f721，均与当前审查文件一致。V2 训练记录、日程、目标、VAL 记录及本轮 training.json 的 SHA 均与相应 completed.json 相符。
- 实际执行以下定向 CPU 测试，输出 **5 passed in 2.07s**；其中混合支持测试有两个参数化用例。未重复 42 项全套：

      env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python -m pytest -p no:cacheprovider -q tests/test_source_update_utility_20261104.py::test_actual_step_double_bce_and_detached_scale_match_independent_formula tests/test_source_update_utility_20261104.py::test_full_utility_preserves_confidence_weighting tests/test_source_update_utility_20261104.py::test_float64_actual_utility_directional_derivative tests/test_source_update_utility_20261104.py::test_mixed_support_averages_only_observed_sensitive_rows

- 额外执行的只读 CPU 合成检查完整命令及结果如下；不写检查点或新测试文件：

      env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/hetaisheng/miniconda3/envs/rl_env/bin/python -c '
      from copy import deepcopy
      import math, torch
      from experiments.risk_aligned_online.redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT
      from experiments.risk_aligned_online.source_anticipation_20261104 import meta, utility
      from experiments.risk_aligned_online.fast_update import normalized_step
      torch.set_num_threads(1); b,_=load_parent(DEFAULT_PARENT); g=torch.Generator().manual_seed(20261104)
      x=b.center+b.scale*.3*torch.randn(2,128,23,generator=g); a=torch.ones_like(x,dtype=torch.bool)
      truth=b.center+b.scale*.7*torch.randn(2,25,23,generator=g); mask=torch.ones_like(truth,dtype=torch.bool); labels=torch.tensor([0.,1.])
      h=deepcopy(b.forecast_head).requires_grad_(True)
      with torch.no_grad(): h.weight.zero_(); h.bias.zero_()
      initial={k:v.detach().clone() for k,v in h.state_dict().items()}
      for mode in (False,True):
       torch.backends.mha.set_fastpath_enabled(mode); _,i=utility.objective(b,h,x,a,x,a,truth,mask,labels,initial_head_state=initial,threshold=.7258946895599365)
       after=meta.features(b,x,a,i["updated_state"])["logit"]
       print("noop",mode,"state",i["updated_state"].detach().abs().max().item(),"logit_gap",(after.detach()-i["baseline_logit"]).abs().max().item(),"utility",i["risk_utility"].item())
      t=.7258946895599365; c=math.log(t/(1-t)); z=torch.full((4,),c-2,dtype=torch.float64); y=torch.tensor([0.,1.,0.,1.],dtype=torch.float64); u=utility._margin_delta(z,z+.05,y,c)/(.05*math.sqrt(1+1e-12))
      print("uniform",u.mean().item(),u[y==0].mean().item(),u[y==1].mean().item())
      s=torch.tensor(1e-9,dtype=torch.float64); scale=.05*torch.sqrt(s*s+1e-12); z=torch.zeros(1,dtype=torch.float64)
      print("curved",(utility._margin_delta(z,z+s*.05+.5*.05**2,torch.zeros(1,dtype=torch.float64),0.)/scale).item())
      v=torch.zeros(2,128,dtype=torch.float64); d=v.clone(); d[:,0]=torch.tensor([1e-6,2.5e-7]); print("epsilon_steps",normalized_step(v,v,d).norm(dim=1).tolist())
      torch.backends.mha.set_fastpath_enabled(False); h=deepcopy(b.forecast_head).requires_grad_(True); initial={k:v.detach().clone() for k,v in h.state_dict().items()}; g=torch.Generator().manual_seed(20261104)
      x=b.center+b.scale*.3*torch.randn(4,128,23,generator=g); q=b.center+b.scale*.3*torch.randn(4,128,23,generator=g); truth=b.center+b.scale*.7*torch.randn(4,25,23,generator=g)
      a=torch.ones_like(x,dtype=torch.bool); m=torch.ones_like(truth,dtype=torch.bool); y=torch.tensor([0.,1.,0.,1.]); _,i=meta.objective(b,h,x,a,q,a,truth,m,y,initial_head_state=initial)
      gb=i["proxy_gradient"].detach(); sb=meta.inner_state(b,i["delta"],i["proxy_gradient"]).detach(); singles=[]; states=[]
      for j in range(4):
       _,k=meta.objective(b,h,x[j:j+1],a[j:j+1],q[j:j+1],a[j:j+1],truth[j:j+1],m[j:j+1],y[j:j+1],initial_head_state=initial)
       singles.append(k["proxy_gradient"].detach()[0]); states.append(meta.inner_state(b,k["delta"],k["proxy_gradient"]).detach()[0])
      print("source_batch","g_norm",gb.norm(dim=1).tolist(),"max_4g_minus_single",(4*gb-torch.stack(singles)).abs().max().item(),"state_gap",(sb-torch.stack(states)).abs().max().item(),"source_grads_none",all(p.grad is None for p in b.parameters()))
      '

      noop False state 0.0 logit_gap 0.0 utility 0.0
      noop True state 0.0 logit_gap 9.5367431640625e-07 utility -0.00019619621562863703
      uniform -0.3781387194019211 0.12186128059783154 -0.8781387194016738
      curved 12503.90049811006
      epsilon_steps [0.03535533901469556, 0.01212678122300048]
      source_batch g_norm [0.0031947714742273092, 0.002678137505427003, 0.0032383594661951065, 0.002701051998883486] max_4g_minus_single 1.1641532182693481e-09 state_gap 6.51925802230835e-09 source_grads_none True

未完成：实际 CUDA 零步等价性、真实样本敏感度／曲率分布、完整流多步效果、EAST 新产物独立核对及留出诊断预测。均未为了补齐本审查启动新实验。无原始信号、HDF、GPU、拟合、git 操作或其他文件写入；审查到此冻结。
