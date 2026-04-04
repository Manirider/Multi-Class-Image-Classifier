# Quick Start Guide

## ⚡ 60-Second Setup

```bash
# 1. Navigate to project
cd multi-class-image-classifier

# 2. Install dependencies (takes ~3-5 min)
pip install -r requirements.txt

# 3. Copy environment variables
cp .env.example .env

# 4. You're ready! Choose your path:
```

---

## 🚀 Path 1: Train & Evaluate (CPU/GPU)

```bash
# Train model (auto-downloads Caltech-101)
python train.py --epochs 5 --batch_size 32

# Evaluate (generates metrics.json)
python evaluate.py

# View results
cat results/metrics.json
```

---

## 🌐 Path 2: Run API Locally

```bash
# Start FastAPI server
uvicorn src.api.main:app --reload

# In another terminal, test:
curl http://localhost:8000/health
# {"status": "ok"}

# Make prediction (after training)
curl -X POST http://localhost:8000/predict \
  -F "file=@image.jpg"
```

---

## 🐳 Path 3: Docker (One Command)

```bash
# Build and run
docker-compose up --build

# In another terminal:
curl http://localhost:8000/health

# Train in container
docker-compose exec api python train.py --epochs 5
```

---

## 📋 Project Structure

```
src/
├── api/         # FastAPI endpoints
├── core/        # Config & logging
├── data/        # Dataset pipeline
├── models/      # ML code
└── utils/       # Transforms

train.py        # Training CLI
evaluate.py     # Evaluation CLI
```

---

## 🔧 Common Commands

```bash
# Validate setup
python validate.py

# Train with custom params
python train.py --epochs 30 --learning_rate 0.0005

# Check logs
tail -f results/training.log

# View API docs
http://localhost:8000/docs
```

---

## 📊 Key Files After Training

```
model/
├── image_classifier.pth     # Trained model
results/
├── metrics.json             # Evaluation metrics
├── training.log             # Training history
```

---

## ❓ Troubleshooting

| Issue | Solution |
|-------|----------|
| Import error | `pip install -r requirements.txt --upgrade` |
| CUDA not found | `python train.py --device cpu` |
| Port 8000 in use | `uvicorn src.api.main:app --port 8001` |
| Dataset not found | First run of `train.py` auto-downloads |

---

## 📖 Full Docs

See **README.md** for:
- Architecture details
- API usage examples
- Performance metrics
- Docker guide
- Future improvements

---

**Status**: Ready to go! 🎉
