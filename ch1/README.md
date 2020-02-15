## 벡터의 내적
- 두 벡터가 얼마나 같은 방향을 향하고 있는가

## 계층 클래스 (layer class)
- Affine 계층, Sigmoid 계층 등 모듈화

### 규칙
1. 모든 계층은 `forward()`와 `backward()` 메소드를 갖는다
2. 모든 계층은 인스턴스 변수인 `params`와 `grads`를 갖는다.
	- params : 가중치, 편향 등 매개변수
	- grads : 기울기 보관

