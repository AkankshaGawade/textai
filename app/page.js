import Image from 'next/image';
import Link from 'next/link';

const Home = () => {
  return (
    <div className="flex justify-between items-center h-screen p-10 bg-slate-100">
      <div className="flex-1 pr-10">
        {/* Image */}
        <Image
          src="/bg.jpeg" // Replace with your image URL
          alt="Social Image"
          width={600}
          height={600}
        />
      </div>
      <div className="flex-1 flex justify-center items-center">
        {/* Signup and Login Buttons */}
        <div className="text-center">
          <Link href="/signup">
            <span className="inline-block bg-blue-500 hover:bg-blue-700 text-white font-bold py-4 px-6 rounded m-3">
              Signup
            </span>
          </Link>
          <Link href="/login">
            <span className="inline-block bg-gray-500 hover:bg-gray-700 text-white font-bold py-4 px-6 rounded m-3">
              Login
            </span>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Home;
