import string
import argparse
import cv2
import queue
import threading
import time


import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate, tensor2im
from model import Model
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelWrapper(Model):

    def __init__(self, opt):
        if 'CTC' in opt.Prediction:
            self.converter = CTCLabelConverter(opt.character)
        else:
            self.converter = AttnLabelConverter(opt.character)
        opt.num_class = len(self.converter.character)
        if opt.rgb:
            opt.input_channel = 3
        super().__init__(opt)
        self = torch.nn.DataParallel(self).to(device)
        print('loading pretrained model from %s' % opt.saved_model)
        self.load_state_dict(torch.load(opt.saved_model, map_location=device))
        self.opt = opt

    def predict(self, img):
        self.eval()
        batch_size = 1
        with torch.no_grad():
            AlignCollate_demo = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD)
            transform_PIL = transforms.ToPILImage()

            image = [transform_PIL(img)]

            image = AlignCollate_demo((image, ""))[0]
            length_for_pred = torch.IntTensor([self.opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, self.opt.batch_max_length + 1).fill_(0).to(device)
            print(image.shape)
            cv2.imshow("", tensor2im(image[0]))

            if 'CTC' in self.opt.Prediction:
                preds = self(image, text_for_pred).log_softmax(2)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = self.converter.decode(preds_index.data, preds_size.data)

            else:
                preds = self(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            pred_max_prob = preds_max_prob[0]
            pred = preds_str[0]
            if 'Attn' in self.opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            return pred, confidence_score


def get_image(cap):
    frame = cap.read()
    #frame = cv2.imread("demo_image/demo_1.png")
    return frame



class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except Queue.Empty:
          pass
      self.q.put(frame)



  def read(self):
    return self.q.get()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')

    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')


    opt = parser.parse_args()
    #model = ModelWrapper(opt)
    model = torch.load("./model.pth")
    #torch.save(model, "./model.pth")
    cap = VideoCapture(0)

    while True:
        img = get_image(cap)
        img = img[180:300, 260:380]
        pred, confidence = model.predict(img)
        print(pred, confidence)
        cv2.imshow("", img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        time.sleep(0.1)
    cap.release()
    cv2.destroyAllWindows()
