import torch
from tqdm import tqdm
import torch.nn as nn

from utils.utils import get_lr

def fit_one_epoch(model_train, model, exp_name,yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, save_period):
    loss        = 0
    val_loss    = 0
    Dehazy_loss = 0
    criterion = nn.MSELoss()

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets, clearimgs = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                    clearimgs = torch.from_numpy(clearimgs).type(torch.FloatTensor).cuda()
                    hazy_and_clear = torch.cat([images, clearimgs], dim=0).cuda()
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                    clearimgs = torch.from_numpy(clearimgs).type(torch.FloatTensor)

            optimizer.zero_grad()

            # outputs         = model_train(images)
            outputs = model_train(hazy_and_clear)
            # outputs = model.forward(hazy_and_clear)
            #
            loss_value_all = 0

            loss_value = yolo_loss(outputs[0], targets)
            # for l in range(len(outputs) - 1):
            #     loss_item = yolo_loss(outputs[l], targets)
            #     loss_value_all  += loss_item
            loss_dehazy = criterion(outputs[1], clearimgs)
            loss_value = 0.2 * loss_value + 0.8 * loss_dehazy


            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            Dehazy_loss += loss_dehazy.item()
            
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1),
                                'Dehazy_loss': Dehazy_loss / (iteration + 1),
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

                optimizer.zero_grad()

                outputs         = model_train(images)


                loss_value = yolo_loss(outputs[0], targets)
                # for l in range(len(outputs)-1):
                #     loss_item = yolo_loss(outputs[l], targets)
                #     loss_value_all  += loss_item
                # loss_value = loss_value_all

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    
    loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), f'logs/{exp_name}/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
