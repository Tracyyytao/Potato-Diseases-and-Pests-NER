import time
from utils import *
from model import *
from config import *


if __name__ == '__main__':
    start_time = time.time()  # 记录开始时间
    dataset = Dataset()
    loader = data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = Model().to(DEVICE)
    fgm = FGM(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for e in range(EPOCH):
        for b, (input, target, mask) in enumerate(loader):

            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)


            y_pred = model(input, mask)

            loss = model.loss_fn(input, target, mask)
            loss.backward()

            fgm.attack()

            loss_adv = model.loss_fn(input, target, mask)
            loss_adv.backward()

            fgm.restore()

            optimizer.step()
            optimizer.zero_grad()

            if b % 10 == 0:
                print('>> epoch:', e, 'loss:', loss.item())

        torch.save(model, MODEL_DIR + f'model_{e}.pth')

    end_time = time.time()  # 记录结束时间
    total_time = end_time - start_time  # 计算总时间
    print(f"Total training time: {total_time} seconds")


