import torch
#from models.PointCloudTransformer.fss_transformer import NeighbourEmbedding, PCT
#from models.PointCloudTransformer.model import PCTSeg

if __name__ == '__main__':
  # test for Neighbour Embedding
  #pc = torch.rand(16, 3, 2048).to('cuda')
  #cls_label = torch.rand(16, 2048).to('cuda')
  #ne = NeighbourEmbedding(in_channel=9).to('cuda')
  #out = ne(pc)
  #print("NeighbourEmbedding output size: ", out.size())

  #pct = PCT(samples=[2048, 2048], in_channel=3).to('cuda')
  #out_pct = pct(pc)
  #print("PCT output size: ", out_pct.size())

  #pct = PCTSeg().to('cuda')
  #out_pct = pct(pc, cls_label)
  #print("PCT output size: ", out_pct.size())

  # from models.encoder.pointnet2_backbone import Pointnet2Backbone
  # backbone_net = Pointnet2Backbone(input_feature_dim=6).cuda()
  # print(backbone_net)
  # backbone_net.eval()
  # pointcloud, features = backbone_net(torch.rand(3, 2048, 9).cuda())
  # print('Input pointcloud shape: ', pointcloud.shape)
  # print('Output features shape: ', features.shape)

  # yannx pointnet/2 model
  from models.Pointnet_Pointnet2_pytorch.models.pointnet2_sem_seg import PointNet2SemSeg
  model = PointNet2SemSeg(13).cuda()
  model.eval()
  out, _ = model(torch.rand(3, 9, 2048).cuda())
  print(out.shape)
 
 
  #from models.pointnet2 import PointNet2SemSegPretrain
  #model = PointNet2SemSegPretrain(num_class=13, input_feature_dim=6).cuda()
  #model.eval()
  #out = model(torch.rand(3, 2048, 9).cuda())
  #print(out.shape)
