import torch

# Third-party packages
from metrics import calculate_metrics

def rotate(img, k, model, device):
    img = torch.rot90(img, k=k, dims=[2, 3]).to(device, dtype=torch.float32)
    logits_rot = model(img).to(device)
    logits_rot = torch.sigmoid(logits_rot)
    
    return logits_rot 

def flip_vertical(img, model, device):
    img = torch.flip(img, dims=[3]).to(device, dtype=torch.float32)
    logits_flipped = model(img).to(device) 
    logits_flipped = torch.sigmoid(logits_flipped)

    return logits_flipped

def flip_horizontal(img, model, device):
    img = torch.flip(img, dims=[1, 2]).to(device, dtype=torch.float32)
    logits_flipped = model(img).to(device) 
    logits_flipped = torch.sigmoid(logits_flipped)

    return logits_flipped

def evaluate(model, data_loader, loss_func, device, threshold, tta, cls_name, average):
    model.eval()

    epoch_targets = []
    epoch_preds = []
    with torch.no_grad():
        epoch_loss = 0
        for image, target in data_loader:
            preds = []
            image, target = image.to(device, dtype=torch.float), target.to(device, dtype=torch.float)

            logits_image = model(image)
            loss = loss_func(logits_image, target)

            logits_image = torch.sigmoid(logits_image)
            
            if tta:
                logits_flipped_H = flip_horizontal(image, model, device)
                preds.extend([logits_image, logits_flipped_H])
            else:
                preds.extend([logits_image])
            
            pred = torch.mean(torch.cat(preds, dim=0), dim=0).unsqueeze(0)
            pred = torch.tensor([torch.argmax(pred).item()])

            epoch_preds.append(pred)
            epoch_targets.append(torch.tensor([torch.argmax(target).item()]))

            epoch_loss += loss.item() * image.size(0)


    epoch_preds = torch.cat(epoch_preds)
    epoch_targets = torch.cat(epoch_targets)

    history = calculate_metrics(epoch_preds, epoch_targets, cls_name, threshold, average)

    return history, epoch_loss / len(data_loader.dataset)

def test(model, data_loader, device, threshold, tta, cls_name, average):
    model.eval()

    epoch_targets = []
    epoch_preds = []
    with torch.no_grad():
        for image, target in data_loader:
            preds = []
            image, target = image.to(device, dtype=torch.float), target.to(device, dtype=torch.float)

            logits_image = model(image)

            logits_image = torch.sigmoid(logits_image)
            
            if tta:
                logits_flipped_H = flip_horizontal(image, model, device)
                preds.extend([logits_image, logits_flipped_H])
            else:
                preds.extend([logits_image])
            
            pred = torch.mean(torch.cat(preds, dim=0), dim=0).unsqueeze(0)

            pred = torch.tensor([torch.argmax(pred).item()])

            epoch_preds.append(pred)
            epoch_targets.append(torch.tensor([torch.argmax(target).item()]))

    epoch_preds = torch.cat(epoch_preds)
    epoch_targets = torch.cat(epoch_targets)


    history = calculate_metrics(epoch_preds, epoch_targets, cls_name, threshold, average)

    return history, epoch_preds, epoch_targets