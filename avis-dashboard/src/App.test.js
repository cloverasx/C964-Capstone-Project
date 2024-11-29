import { render, screen } from '@testing-library/react';
import App from './App';

describe('AVIS Application', () => {
  test('renders main dashboard', () => {
    render(<App />);
    const headerElement = screen.getByText(/Automated Vehicle Identification System/i);
    expect(headerElement).toBeInTheDocument();
  });

  test('renders upload section', () => {
    render(<App />);
    const uploadSection = screen.getByText(/Upload Image/i);
    expect(uploadSection).toBeInTheDocument();
  });

  test('renders URL input section', () => {
    render(<App />);
    const urlSection = screen.getByText(/Image URL/i);
    expect(urlSection).toBeInTheDocument();
  });
});
