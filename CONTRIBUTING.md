# Contributing to Highway Guardian

Chúng tôi hoan nghênh mọi đóng góp cho dự án Highway Guardian! Dưới đây là hướng dẫn chi tiết về cách bạn có thể đóng góp.

## Cách Đóng góp

### 1. Báo cáo Lỗi (Bug Reports)

Nếu bạn phát hiện lỗi, vui lòng tạo issue với thông tin sau:
- Mô tả chi tiết lỗi
- Các bước để tái tạo lỗi
- Kết quả mong đợi vs kết quả thực tế
- Screenshots (nếu có)
- Thông tin môi trường (OS, Python version, etc.)

### 2. Đề xuất Tính năng (Feature Requests)

Để đề xuất tính năng mới:
- Mô tả tính năng và lý do cần thiết
- Đưa ra ví dụ sử dụng
- Thảo luận về implementation approach

### 3. Code Contributions

#### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/VThanhNguyen2002/highway-guardian.git
cd highway-guardian

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

#### Development Workflow

1. **Fork repository** và clone về máy local
2. **Tạo branch mới** cho feature/bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Implement changes** với coding standards
4. **Write tests** cho code mới
5. **Run tests** để đảm bảo không break existing functionality
6. **Commit changes** với clear commit messages
7. **Push branch** và tạo Pull Request

#### Coding Standards

- **Python Style**: Follow PEP 8
- **Formatting**: Use Black formatter
- **Linting**: Use flake8
- **Type Hints**: Use type annotations
- **Docstrings**: Follow Google style

```python
def example_function(param1: str, param2: int) -> bool:
    """Example function with proper documentation.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input is provided
    """
    pass
```

#### Testing

- Write unit tests cho mọi function/class mới
- Maintain test coverage > 80%
- Use pytest framework
- Test cả positive và negative cases

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_detection.py::test_traffic_sign_detection
```

#### Documentation

- Update README.md nếu cần
- Add docstrings cho functions/classes
- Update world.md cho theoretical changes
- Add examples trong docs/ folder

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up-to-date with main

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots (if applicable)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## Development Setup với Docker

```bash
# Build development image
docker-compose -f docker-compose.dev.yml build

# Start development environment
docker-compose -f docker-compose.dev.yml up

# Run tests in container
docker-compose -f docker-compose.dev.yml exec app pytest
```

## Code Review Process

1. **Automated Checks**: CI/CD pipeline runs tests
2. **Peer Review**: At least 1 reviewer approval required
3. **Maintainer Review**: Final review by project maintainers
4. **Merge**: Squash and merge to main branch

## Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `priority-high`: High priority issue
- `priority-low`: Low priority issue

## Communication

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For general questions
- **Email**: [your-email@example.com] for private matters

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given appropriate GitHub repository permissions

## Getting Help

Nếu bạn cần hỗ trợ:
1. Check existing issues và documentation
2. Search GitHub Discussions
3. Create new issue với label "help wanted"
4. Contact maintainers directly

## Code of Conduct

### Our Pledge

Chúng tôi cam kết tạo môi trường thân thiện, chào đón mọi người bất kể:
- Tuổi tác, giới tính, bản dạng giới
- Khuyết tật, ngoại hình
- Dân tộc, quốc tịch
- Tôn giáo, chính trị
- Kinh nghiệm, trình độ

### Expected Behavior

- Sử dụng ngôn ngữ thân thiện và inclusive
- Tôn trọng quan điểm và kinh nghiệm khác nhau
- Chấp nhận constructive criticism
- Focus vào điều tốt nhất cho community
- Thể hiện empathy với community members

### Unacceptable Behavior

- Ngôn ngữ hoặc hình ảnh sexual
- Trolling, insulting, hoặc derogatory comments
- Harassment công khai hoặc riêng tư
- Publishing thông tin cá nhân của người khác
- Conduct không professional khác

## License

Bằng cách đóng góp, bạn đồng ý rằng contributions sẽ được licensed dưới MIT License.

---

Cảm ơn bạn đã quan tâm đến việc đóng góp cho Highway Guardian! 🚗🚦