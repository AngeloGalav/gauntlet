import 'bootstrap/dist/css/bootstrap.css';
import MyFooter from './components/Footer';
import ImageProcessingPage from './components/ImageProcessingPage';
import Navbar from './components/NavBar';
import './App.css';

function App() {
  return (
    <div>
    <Navbar />
    <div
    className='min-vh-90' style={{minHeight:"100%", maxWidth:"100%"}}
    >
      <ImageProcessingPage />
      <MyFooter />
    </div>
    </div>
  );
}

export default App;
