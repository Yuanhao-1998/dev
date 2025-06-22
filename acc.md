 
sum(p.numel() for p in model.backbone.body.layer4.parameters())
sum(p.numel() for p in model.parameters() if p.requires_grad)
model = torch.load("c:/projects/oil-level-prediction/output/oil_container_mobilenet_final.pt", weights_only=False)
model.eval().to(device)
TODO: 情未了
No problem, but addtional code needs to be written (usually when you are skipping something)
标识处有功能代码待编写，待实现的功能在说明中会简略说明。
FIXME: Fuck Me
This works, sort of, but it could be done better. (usually code written in a hurry that needs rewriting)
标识处代码需要修正，甚至代码是错误的，不能工作，需要修复，如何修正会在说明中简略说明。
XXX: 陷阱
Warning about possible pitfalls, can be used as NOTE:XXX:
标识处代码虽然实现了功能，但是实现的方法有待商榷，希望将来能改进，要改进的地方会在说明中简略说明。
NOTE: 汝等凡人
Description of how the code works (when it isn’t self evident)
HACK: 补锅踩雷填坑
Not very well written or malformed code to circumvent a problem/bug. Should be used as HACK:FIXME
BUG: 丢锅埋雷挖坑
There is a problem here.
