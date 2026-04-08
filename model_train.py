import torch
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn 
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),      
        transforms.ToTensor(),              
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])   
    ])

    data = ImageFolder(root=r"data\mad\archive(3)\imagery", transform=transform)

    train_size = int(0.8 * len(data))
    val_size = int(len(data) - train_size)
    train_data, val_data = random_split(data, [train_size, val_size])
    batch_size = 32
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=8 )
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, num_workers=8 )

    model = efficientnet_b0(weights = None)
    model.classifier[1] = nn.Linear(1280, 3)
    model.load_state_dict(torch.load(r"efficientnet_best.pth", weights_only=True))
    loss_fn = torch.nn.CrossEntropyLoss()
    lr_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("Начало обучения...")
    best_acc = 0.0  
    for epoch in range(20):

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
    
        for images, labels in train_loader:


            pred = model(images)
            loss = loss_fn(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = pred.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():                     
            for images, labels in val_loader:

                pred = model(images)
                loss = loss_fn(pred, labels)
                val_loss += loss.item()

                _, predicted = pred.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

    
        print(f"Эпоха {epoch+1:2d}/20")
        print(f"   TRAIN  Loss: {train_loss:.4f} | Acc: {train_acc:6.2f}%")
        print(f"   VAL   Loss: {val_loss:.4f} | Acc: {val_acc:6.2f}%")
        print("-" * 50)

   
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "efficientnet_best1.pth")
            print(f"Val Acc = {val_acc:.2f}%")

    print("Обучение завершено!")
    print(f" точность на валидации: {best_acc:.2f}%")
    print(" модель сохранена : efficientnet_best.pth")