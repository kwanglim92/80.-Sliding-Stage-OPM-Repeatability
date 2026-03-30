# Sliding Stage OPM Repeatability Analyzer

## Project Context
- **Framework**: PySide6 + Matplotlib (Catppuccin Mocha dark theme)
- **Domain**: Park Systems XE 장비의 Sliding Stage OPM 반복성 분석
- **Data**: TIFF 프로파일 (Tag 50434), 9-point wafer grid (1_LT ~ 9_RB)
- **Entry point**: `main.py` → `src/` 패키지

## Development Rules
코드를 작성하거나 수정하기 전에 반드시 `.agent/rules.md`를 읽고 따를 것.

핵심 규칙 요약:
1. 코딩 전 구현 계획을 먼저 세울 것
2. 요청하지 않은 추가 기능/추상화/에러 처리를 임의로 추가하지 말 것
3. 지시받은 목표 지점의 코드만 수정할 것
4. 구체적이고 검증 가능한 목표에 집중할 것
5. **Skill-First**: 코드 작성 전 `.agent/skills/registry.json`에서 관련 skill을 검색하고, 해당 SKILL.md의 패턴을 따를 것

## Skills Reference
- **Registry**: `.agent/skills/registry.json` (17개 skill, tags/dependencies/source_files 포함)
- **한국어 skill**: `.agent/skills/skills-ko/{폴더명}/SKILL.md`
- **영문 skill**: `.agent/skills/skills-en/{폴더명}/SKILL.md`

주요 skill:
| ID | Skill | 이 프로젝트 관련도 |
|----|-------|-------------------|
| 01 | 다크테마 GUI | COLORS dict, dark theme 패턴 |
| 02 | 정적 차트 (Matplotlib) | plot_manager.py 차트 생성 |
| 07 | 통계 엔진 | analyzer.py 분석 로직 |
| 11 | 커스텀 위젯 | UI 컴포넌트 패턴 |
| 14 | 설정 관리 | 설정 저장/로드 |
| 16 | 검증 QA | 작업 완료 후 검증 체크리스트 |
