AI Bootcamp 최종 과제 안내


1. 과제 주제
나만의 End-to-End AI Agent 서비스 개발


2. 과제 개요
이번 과제는 Bootcamp에서 학습한 기술들을 기반으로, 실제 사용 가능한 AI Agent 서비스를 직접 설계하고 구현하는 것을 목표로 합니다.

수강생은 문제 정의부터 프롬프트 설계, Agent 구조 설계, RAG 기반 지식 결합, 사용자 경험 구현까지 하나의 흐름으로 연결된 완결형 서비스를 개발해야 합니다.

Agent의 핵심 기능은 LangChain/LangGraph 기반으로 구현하며, 서비스 수준을 높이기 위해 필요에 따라 출력 구조화, 도구·시스템 연동, Agent 간 협업 등 다양한 확장 기능을 적용할 수 있습니다.

최종 결과물은 단순 예제를 넘어 실제 업무·서비스 환경에서도 활용 가능한 수준의 실무형 AI Agent여야 합니다.



3. 과제 목표
Prompt → 설계 → 구현 → 패키징까지 Bootcamp 전 과정 기반 Agentic 서비스를 완성하는 것을 목표로 합니다.

LangChain/LangGraph 기반 역할 기반 또는 Multi-Agent 구조 설계

RAG 기반 지식 응답, Structured Output·Function Calling 등 고도화 기법 기반 안정적 응답 구성

UI/서비스로 패키징해 실제 사용 가능한 형태로 구현, 최신 기술(MCP/A2A 등)을 적용해 확장성·완성도 강화

※ 핵심은 모든 기술을 다 쓰는 것이 아닌, 기술을 적절히 선택·조합해 높은 품질의 Agent 서비스를 설계·구현하는 것입니다.



4. 수행 가이드라인


4.1 주제 선정
실제 적용 가능성이 높고, 해결하고자 하는 문제가 명확한 주제를 선정합니다.

기존 서비스와 차별화할 요소를 고려하면 설계 및 구현 완성도 향상에 도움이 됩니다.



4.2 필수 기술 요소
대부분의 완결적 Agent 서비스에 공통적으로 요구되는 하기 내용이 포함되어야 합니다. (Bootcamp 학습 내용)



1) Prompt Engineering

역할 기반 프롬프트 설계, CoT, Few-shot 등 고품질 응답을 위한 구성

다양한 입력 상황에서도 일관성을 확보하는 프롬프트 구조화



2) LangChain/LangGraph 기반 Agent 구현

Multi-Agent 구조 설계 (단일 Agent 미인정)

Tool Calling, ReAct 기반 실행, Memory 활용



3) RAG (Retrieval-Augmented Generation)

데이터 전처리, 임베딩, Vector DB (FAISS/Chroma 등) 구성

검색 기반 지식 보강 기능 설계 및 구현



4) 서비스 개발 및 패키징

Streamlit 또는 원하는 프론트엔드 프레임워크로 UI 구성

FastAPI 기반 백엔드 구성, Docker 기반 배포 환경 구성(선택)



4.3 선택 요소 (Advanced Option)


A. LLM Fundamentals 기반 고도 설계

Structured Output, Function Calling, Reasoning 흐름 설계



B. MCP(Model Context Protocol) 기반 도구 연결

파일 시스템 접근, 외부 API/사내 시스템 연동 구조 설계



C. A2A(Agent-to-Agent) 협업 구조 설계

역할별 Agent 간 통신, 협업 기반 문제 해결 구조



(기타 유의사항)
환경변수로 API Key를 안전하게 관리해야 하며, 파일 모듈화 등 실무 개발 요소도 평가에 반영됩니다.

실습 코드를 그대로 사용하거나 단순 변형한 경우 인정되지 않으며, 단순히 구동 가능한 프로그램이 아닌 실제 활용 가능한 수준의 서비스를 구현해야 합니다.